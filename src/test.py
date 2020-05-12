import torch
from argument import test_args
from loader import get_loader
from model import create_model
import pandas as pd
import os

if __name__ == '__main__':
    args = test_args()
    test_loader = get_loader(args, 'test', shuffle=False)

    model = create_model(args)
    model.load_model(args.model_path)
    model.eval()

    phonome_pred = []
    test_num = 0
    with torch.no_grad():
        for i, (x_padded, x_lens) in enumerate(test_loader):
            x_padded = x_padded.to(args.device)
            output, output_lens = model(x_padded, x_lens, is_training=False)
            phonome_preds = model.decoder.decode(output, output_lens)
            phonome_pred.extend(phonome_preds)
            test_num += len(x_lens)

    d = {'Id': list(range(test_num)), 'Predicted': phonome_pred}
    df = pd.DataFrame(data=d)
    df.to_csv(os.path.split(args.model_path)[0] + '/' + args.result_file, header=True, index=False)
    print('Testing is done, result is saved')
