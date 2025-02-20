import os
import argparse
import shutil
import numpy as np

from tqdm import tqdm
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from factory.survival import make_surv_array, surv_likelihood
from dataset.SurvivalDataset import SurvivalDataset
from models.resnet1d import ResNet18, ResNet34, ResNet50


def save_ckpt(state, is_best, model_save_dir):
    model_save_dir = Path(model_save_dir)
    model_save_dir.mkdir(parents=True, exist_ok=True)
    current_w = os.path.join(model_save_dir, 'current.pth')
    best_w = os.path.join(model_save_dir, 'best.pth')
    torch.save(state, current_w)
    if is_best: shutil.copyfile(current_w, best_w)


def train(dataloader, model, criterion, scheduler, optimizer, device):
    model.train()
    train_epoch_loss = []
    with tqdm(dataloader, ncols=150, desc='Train') as tbar:
        for data, labels in tbar:
            data, labels = data.to(device), labels.to(device)

            output = model(data)
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_epoch_loss.append(loss.item())
            tbar.set_postfix(loss=loss.item())
            tbar.update()

    scheduler.step()
    avg_train_loss = np.average(train_epoch_loss)
    print('Loss: %.4f' % avg_train_loss)
    return avg_train_loss


def evaluate(dataloader, model, criterion, device):
    model.eval()
    val_epoch_loss = []
    with tqdm(dataloader, ncols=100, desc='Validate') as tbar:
        for data, labels in tbar:
            data, labels = data.to(device), labels.to(device)
            output = model(data)
            loss = criterion(output, labels)
            val_epoch_loss.append(loss.item())
            tbar.set_postfix(loss=loss.item())
            tbar.update()

    avg_val_loss = np.average(val_epoch_loss)

    print('Loss: %.4f' % avg_val_loss)
    return avg_val_loss


def main(model, label, train_csv_path, val_csv_path):
    args.best_metric = 100
    args.best_epoch = 1

    if args.use_gpu and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.device_ids[0]}')
    else:
        device = 'cpu'
    train_dataset = SurvivalDataset('train', train_csv_path, label=label, h5_path=args.h5_path,
                                    h5_csv_path=args.h5_csv_path)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    val_dataset = SurvivalDataset('validate', val_csv_path, label=label, h5_path=args.h5_path,
                                  h5_csv_path=args.h5_csv_path)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = torch.nn.DataParallel(model, device_ids=args.device_ids)
    model = model.cuda(device=args.device_ids[0])

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)
    criterion = surv_likelihood(n_intervals)

    for epoch in range(1, args.epochs + 1):
        avg_train_loss = train(train_loader, model, criterion, scheduler, optimizer, device)
        avg_valid_loss = evaluate(val_loader, model, criterion, device)

        state = {"state_dict": model.state_dict()}
        save_ckpt(state, avg_valid_loss < args.best_metric, args.ckpt)
        if avg_valid_loss < args.best_metric:
            args.best_metric = avg_valid_loss
            args.best_epoch = epoch
            print(f"Save best:{args.best_epoch}---{args.best_metric}")

        print(f"Epoch {epoch} Train_loss {avg_train_loss}  Val_loss {avg_valid_loss}\n-------------------------------")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', '--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--use-gpu', default=True, action='store_true', help='Use GPU')
    parser.add_argument('--device-ids', type=list, default=[0], help='Can use device ids')
    parser.add_argument('--ckpt', type=str, default='checkpoints/', help='Path to saved model')
    parser.add_argument('--h5-path', type=str, default='data/ecg_data.h5', help='H5 file path')
    parser.add_argument('--h5-csv-path', type=str, default='data/ecg_labels.csv', help='H5_csv file path')
    parser.add_argument('--dat-dir', type=str, default='', help='Dat folder path')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    breaks = np.arange(0., 365. * 5, 365. / 8)
    n_intervals = len(breaks) - 1
    model = ResNet34(num_classes=n_intervals)
    main(model=model,
         label='fu_label',
         train_csv_path='dataset/VHD/train_VHD.csv',
         val_csv_path='dataset/VHD/val_VHD.csv')
