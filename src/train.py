import numpy as np
import torch
from tqdm import tqdm
from argument import train_args
import os
from loader import get_loader
from model import create_model


def save_log(message, args):
    log_name = os.path.join(args.expr_dir, 'loss_log.txt')
    with open(log_name, "a") as log_file:
        log_file.write('\n' + message)


def validate(args, model, loader):
    model.eval()
    running_batch, running_loss, running_dist = 0, 0.0, 0.0
    with torch.no_grad():
        for i, (x_padded, x_lens, y_padded, y_lens) in enumerate(loader):
            x_padded, y_padded = x_padded.to(args.device), y_padded.to(args.device)
            output, output_lens, loss = model(x_padded, x_lens, y_padded, y_lens)
            dist = model.get_edit_dist(output, output_lens, y_padded, y_lens)
            running_dist += dist
            running_batch += len(x_lens)
            running_loss += loss.item()

            del x_padded
            del y_padded
            del x_lens
            del y_lens
            del dist
            del output
            del output_lens
            del loss

        running_loss /= running_batch
        running_dist /= running_batch
    return running_loss, running_dist


if __name__ == '__main__':
    args = train_args()

    train_loader = get_loader(args, 'train', shuffle=True)
    val_loader = get_loader(args, 'val', shuffle=False)
    args.loader_length = len(train_loader)

    model = create_model(args)
    model.train_setup()

    from shutil import copyfile
    copyfile('model.py', os.path.join(args.expr_dir, 'model.py'))

    cur_dist = np.inf
    pbar = tqdm(range(1, args.num_epochs + 1), ncols=0)
    for epoch in pbar:
        model.train()
        if args.use_step_schedule:
            model.update_learning_rate()
        running_batch, running_loss, running_dist = 0, 0.0, 0.0
        for i, (x_padded, x_lens, y_padded, y_lens) in enumerate(train_loader):
            if not args.use_step_schedule and not args.use_reduce_schedule:
                model.update_learning_rate()
            model.optimize_parameters(x_padded, x_lens, y_padded, y_lens)
            states = model.get_current_states()
            running_batch += len(x_lens)
            for name, value in states.items():
                if name == 'loss':
                    running_loss += value
                elif name == 'edit_dist':
                    running_dist += value
                elif name == 'lr':
                    lr = value
            if (i + 1) % args.check_step == 0:  # print every 5 mini-batches
                message = '[%d, %5d] lr: %.5f loss: %.5f edit_distance: %.3f' % \
                          (epoch, i + 1, lr, running_loss/running_batch, running_dist/running_batch)
                pbar.set_description(desc=message)
                save_log(message, args)
        if args.use_reduce_schedule:
            model.update_learning_rate(running_dist/running_batch)

        if epoch % args.eval_step == 0:
            model.eval()
            val_loss, val_dist = validate(args, model, val_loader)
            message = 'Epoch: %d, val Loss %.5f, val edit-distance: %.3f' % (epoch, val_loss, val_dist)
            pbar.set_description(desc=message)
            save_log(message, args)
            message = '-' * 20
            save_log(message, args)
            if val_dist < cur_dist:
                model.save_model(epoch)
                cur_dist = val_dist
