from .utils import accuracy, load_data
from .player import Planner
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def train(args):
    model = Planner()

    """
    Your code here

    """
    import torch

    device = torch.device('cpu')

    model.to(device)
    if args.continue_training:
        from os import path
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % args.model)))

    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)

    train_data = load_data('dataset1')
    # valid_data = load_data('data/valid')

    for epoch in range(args.num_epoch):
        model.train()
        loss_vals, acc_vals, vacc_vals = [], [], []
        for img, label in train_data:
            img, label = img.to(device), label.to(device)

            logit = model(img)
            print(logit.shape())
            print(label.shape())
            loss_val = F.cross_entropy(logit, label)
            acc_val = accuracy(logit, label)

            loss_vals.append(loss_val.detach().cpu().numpy())
            acc_vals.append(acc_val.detach().cpu().numpy())

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

        avg_loss = sum(loss_vals) / len(loss_vals)
        avg_acc = sum(acc_vals) / len(acc_vals)

        model.eval()
        # for img, label in valid_data:
        #     img, label = img.to(device), label.to(device)
        #     vacc_vals.append(accuracy(model(img), label).detach().cpu().numpy())
        # avg_vacc = sum(vacc_vals) / len(vacc_vals)

        print('epoch %-3d \t loss = %0.3f \t acc = %0.3f \t val acc = %0.3f' % (epoch, avg_loss, avg_acc, avg_vacc))
    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', choices=['linear', 'mlp'], default='linear')
    # Put custom arguments here
    parser.add_argument('-n', '--num_epoch', type=int, default=50)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-c', '--continue_training', action='store_true')

    args = parser.parse_args()
    train(args)
