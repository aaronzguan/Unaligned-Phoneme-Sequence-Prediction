import argparse
import torch
import datetime
import os


def common_args():
    parser = argparse.ArgumentParser(description='Speech Recognition')
    # parser.add_argument('--_data_root', default='/home/ubuntu/data/hw3p2/', type=str, help='dataset root path')
    parser.add_argument('--_data_root', default='/Users/aaron/Desktop/11785-Intro to Deep Learning/Homework Part2/Homework3/data/', type=str, help='dataset root path')
    parser.add_argument('--_train_data', default='wsj0_train', type=str, help='validation dataset')
    parser.add_argument('--_train_label', default='wsj0_train_merged_labels.npy', type=str, help='validation labels')
    parser.add_argument('--_val_data', default='wsj0_dev.npy', type=str, help='validation dataset')
    parser.add_argument('--_val_label', default='wsj0_dev_merged_labels.npy', type=str, help='validation labels')
    parser.add_argument('--_test_data', default='wsj0_test', type=str, help='test dataset')
    ##
    parser.add_argument('--checkpoints_dir', default='../checkpoints/', help='checkpoint folder root')
    parser.add_argument('--gpu_ids', default='0', type=str, help='gpu ids: e.g. 0 0,1,2, 0,2.')
    parser.add_argument('--beam_width', default=20, type=int, help='beam width for beam ctc decode')
    #
    parser.add_argument('--use_pca', default=False, type=bool, help='use pca for data preprocess?')
    parser.add_argument('--pca_components', default=20, type=int, help='number of pca components')
    parser.add_argument('--data_aug', default=False, type=bool, help='use augmentation for data preprocess?')
    ##
    parser.add_argument('--num_layers', default=4, type=int, help='number of stacked rnn layers')
    parser.add_argument('--feature_size', default=40, type=int, help='input feature dimension')
    parser.add_argument('--hidden_size', default=512, type=int, help='hidden unit of rnn cell')
    parser.add_argument('--vocab_size', default=47, type=int, help='output vocab dimension')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    return parser


def train_args():
    parser = common_args()
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay for optimizer')
    parser.add_argument('--lr_gamma', default=0.5, type=float, help='learning rate decay gamma')
    parser.add_argument('--num_epochs', default=30, type=int, help='epoch number')
    parser.add_argument('--use_reduce_schedule', default=False, type=bool, help='use reduce on plateau scheduler?')
    parser.add_argument('--use_step_schedule', default=True, type=bool, help='use step scheduler?')
    parser.add_argument('--decay_steps', default='11, 15, 20, 24, 28', type=str, help='step where learning rate decay by 0.1')
    parser.add_argument('--check_step', default=5, type=int, help='step for check batch-loss in one epoch')
    parser.add_argument('--eval_step', default=1, type=int, help='step for validation')
    args = modify_args(parser, dev=False)
    return args


def test_args():
    parser = common_args()
    parser.add_argument('--model_path', default='/Users/aaron/Desktop/11785-Intro to Deep Learning/Homework Part2/Homework3/checkpoints/20200327-214513/13_net.pth', type=str)
    # parser.add_argument('--model_path', default='/home/ubuntu/HomeworkPart2/Homework3/checkpoints/20200329-160955/17_net.pth', type=str)
    parser.add_argument('--result_file', default='hw3p2_phonome_result.csv', type=str)
    args = modify_args(parser, dev=True)
    return args


def modify_args(parser, dev):
    args = parser.parse_args()
    ## Set gpu ids
    if not torch.cuda.is_available():
        args.gpu_ids = None
    else:
        str_ids = args.gpu_ids.split(',')
        args.gpu_ids = []
        for str_id in str_ids:
            gpu_id = int(str_id)
            if gpu_id >= 0:
                args.gpu_ids.append(gpu_id)
        if len(args.gpu_ids) > 0:
            torch.cuda.set_device(args.gpu_ids[0])
    ## Set device
    args.device = torch.device('cuda:{}'.format(args.gpu_ids[0])) if args.gpu_ids else torch.device('cpu')
    if not dev:
        ## Set decay steps
        str_steps = args.decay_steps.split(',')
        args.decay_steps = []
        for str_step in str_steps:
            str_step = int(str_step)
            args.decay_steps.append(str_step)
        ## Set names
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        args.name = '{}'.format(current_time)
    ## Print Options
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(args).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)
    if not dev:
        ## Save Options
        args.expr_dir = os.path.join(args.checkpoints_dir, args.name)
        os.makedirs(args.expr_dir)
        file_name = os.path.join(args.expr_dir, 'arguments.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')
    return args
