import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
import os
from collections import OrderedDict
from decoder import BeamCTCDecoder
from phoneme_list import PHONEME_MAP


def scale_cos(x):
    start = 1e-3
    end = 1e-5
    return start + (1 + np.cos(np.pi * (1 - x))) * (end - start) / 2


class ParamScheduler:
    def __init__(self, optimizer, scale_fn, total_steps):
        self.optimizer = optimizer
        self.scale_fn = scale_fn
        self.total_steps = total_steps
        self.current_iteration = 0

    def step(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.scale_fn(self.current_iteration / self.total_steps)

        self.current_iteration += 1


def weights_init(m, type='xavier'):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1 or classname.find('Conv2d') != -1 or classname.find('Conv1d') != -1:
        if type == 'xavier':
            nn.init.xavier_normal_(m.weight)
        elif type == 'kaiming':
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif type == 'orthogonal':
            nn.init.orthogonal_(m.weight)
        elif type == 'gaussian':
            m.weight.data.normal_(0, 0.01)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class SpeechNet(nn.Module):
    def __init__(self, args):
        super(SpeechNet, self).__init__()
        if args.use_pca:
            input_size = args.pca_components
        elif args.data_aug:
            input_size = args.feature_size * 2
        else:
            input_size = args.feature_size

        self.cnn = nn.Sequential(nn.Conv1d(input_size, 128, kernel_size=3, padding=1),
                                 nn.ReLU(inplace=True),
                                 nn.BatchNorm1d(128),
                                 nn.Conv1d(128, 128, kernel_size=3, padding=1),
                                 nn.ReLU(inplace=True),
                                 nn.BatchNorm1d(128),
                                 nn.Conv1d(128, 256, kernel_size=3, padding=1),
                                 nn.ReLU(inplace=True),
                                 nn.BatchNorm1d(256),
                                 nn.Conv1d(256, 256, kernel_size=3, padding=1),
                                 nn.ReLU(inplace=True),
                                 nn.BatchNorm1d(256),
                                 nn.Conv1d(256, 256, kernel_size=3, padding=1),
                                 nn.ReLU(inplace=True),
                                 nn.BatchNorm1d(256),
                                 )

        self.lstm = nn.LSTM(256, args.hidden_size, args.num_layers, batch_first=True, bidirectional=True)

        self.classifier = nn.Sequential(nn.Linear(args.hidden_size * 2, args.hidden_size * 2),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(args.hidden_size*2, args.vocab_size)
                                        )

    def forward(self, x_padded, x_lens):
        x_padded = self.cnn(x_padded.permute(0, 2, 1)).permute(0, 2, 1)
        x_packed = pack_padded_sequence(x_padded, x_lens, batch_first=True, enforce_sorted=False)
        output_packed, _ = self.lstm(x_packed)
        output_padded, output_lens = pad_packed_sequence(output_packed, batch_first=True)
        # Log softmax after output layer is required since`nn.CTCLoss` expects log probabilities.
        output = self.classifier(output_padded).log_softmax(2)
        return output, output_lens


class create_model(nn.Module):
    def __init__(self, args):
        super(create_model, self).__init__()
        self.args = args
        self.model = SpeechNet(args)
        self.model.to(args.device)
        self.criterion = nn.CTCLoss()
        self.decoder = BeamCTCDecoder(PHONEME_MAP, blank_index=0, beam_width=args.beam_width)

        self.state_names = ['loss', 'edit_dist', 'lr']

    def train_setup(self):
        self.lr = self.args.lr
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        if self.args.use_step_schedule:
            self.scheduler = MultiStepLR(self.optimizer, milestones=self.args.decay_steps, gamma=self.args.lr_gamma)
        elif self.args.use_reduce_schedule:
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=1)
        else:
            self.scheduler = ParamScheduler(self.optimizer, scale_cos, self.args.num_epochs * self.args.loader_length)
#         self.model.apply(weights_init)
        self.model.train()

    def optimize_parameters(self, input, input_lens, target, target_lens):
        input, target = input.to(self.args.device), target.to(self.args.device)
        output, output_lens, self.loss = self.forward(input, input_lens, target, target_lens)

        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

        self.edit_dist = self.get_edit_dist(output, output_lens, target, target_lens)

        del input
        del target
        del input_lens
        del target_lens
        del output
        del output_lens

    def update_learning_rate(self, dist=None):
        if self.args.use_reduce_schedule:
            self.scheduler.step(dist)
        else:
            self.scheduler.step()
        self.lr = self.optimizer.param_groups[0]['lr']

    def get_current_states(self):
        errors_ret = OrderedDict()
        for name in self.state_names:
            if isinstance(name, str):
                # float(...) works for both scalar tensor and float number
                errors_ret[name] = float(getattr(self, name))
        return errors_ret

    def get_edit_dist(self, output, output_lens, target, target_lens):
        output, target = output.cpu(), target.cpu()
        phonome_preds = self.decoder.decode(output, output_lens)
        phonomes = self.decoder.convert_to_strings(target, target_lens)
        edit_dist = np.sum(
            [self.decoder.Lev_dist(phonome_pred, phonome) for (phonome_pred, phonome) in zip(phonome_preds, phonomes)])
        return edit_dist

    def forward(self, input, input_lens, target=None, target_lens=None, is_training=True):
        output, output_lens = self.model(input, input_lens)
        if is_training:
            # The official documentation is your best friend: https://pytorch.org/docs/stable/nn.html#ctcloss
            # nn.CTCLoss takes 4 arguments to compute the loss:
            # [log_probs]: Prediction of your model at each time step. Shape: (seq_len, batch_size, vocab_size)
            # Values must be log probabilities. Neither probabilities nor logits will work.
            # Make sure the output of your network is log probabilities, by adding a nn.LogSoftmax after the last layer.
            # [targets]: The ground truth sequences. Shape: (batch_size, seq_len)
            # Values are indices of phonemes. Again, remember that index 0 is reserved for "blank"
            # [input_lengths]: Lengths of sequences in log_probs. Shape: (batch_size,).
            # This is not necessarily the same as lengths of input of the model.
            # [target_lengths]: Lengths of sequences in targets. Shape: (batch_size,).
            loss = self.criterion(output.permute(1, 0, 2), target, input_lens, target_lens)
            return output, output_lens, loss
        else:
            return output, output_lens,

    def train(self):
        try:
            self.model.train()
        except:
            print('train() cannot be implemented as model does not exist.')

    def eval(self):
        try:
            self.model.eval()
        except:
            print('eval() cannot be implemented as model does not exist.')

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))

    def save_model(self, which_epoch):
        save_filename = '%s_net.pth' % (which_epoch)
        save_path = os.path.join(self.args.expr_dir, save_filename)
        if torch.cuda.is_available():
            try:
                torch.save(self.model.module.cpu().state_dict(), save_path)
            except:
                torch.save(self.model.cpu().state_dict(), save_path)
        else:
            torch.save(self.model.cpu().state_dict(), save_path)

        self.model.to(self.args.device)