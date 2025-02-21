import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score

from dataset.SurvivalDataset import SurvivalDataset
from factory.survival import make_surv_array, surv_likelihood, pred_surv
from models.resnet1d import ResNet18, ResNet34, ResNet50


def eval_batch(model, label_csv, label_name, checkpoint_dir, h5_path, h5_csv_path):
    df_test = pd.read_csv(label_csv, low_memory=False)
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    model = model.cuda(device=device_ids[0])
    param_dict = torch.load(f'{checkpoint_dir}/best.pth', map_location=f'cuda:{device_ids[0]}')
    model.load_state_dict(param_dict['state_dict'])
    model.eval()

    data_dataset = SurvivalDataset('test', label_csv, label=label_name, h5_path=h5_path, h5_csv_path=h5_csv_path)
    data_loader = DataLoader(data_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=False)

    output_list = []
    for ecg_data, label in tqdm(data_loader):
        ecg_data, label = ecg_data.cuda(device=device_ids[0]), label.cuda(device=device_ids[0])
        output = model(ecg_data)
        survival_probs = output.cpu().data.numpy()
        output_list.append(survival_probs)

    y_preds = np.vstack(output_list)
    risk_scores_1 = []
    risk_scores_2 = []
    risk_scores_3 = []
    risk_scores_4 = []
    risk_scores_5 = []
    for i in range(len(df_test)):
        risk_scores_1.append(pred_surv(y_preds[i:i + 1], breaks, 365. * 1)[0])
        risk_scores_2.append(pred_surv(y_preds[i:i + 1], breaks, 365. * 2)[0])
        risk_scores_3.append(pred_surv(y_preds[i:i + 1], breaks, 365. * 3)[0])
        risk_scores_4.append(pred_surv(y_preds[i:i + 1], breaks, 365. * 4)[0])
        risk_scores_5.append(pred_surv(y_preds[i:i + 1], breaks, 365. * 5)[0])
    df_test['risk_scores_1'] = risk_scores_1
    df_test['risk_scores_2'] = risk_scores_2
    df_test['risk_scores_3'] = risk_scores_3
    df_test['risk_scores_4'] = risk_scores_4
    df_test['risk_scores_5'] = risk_scores_5

    return df_test


def plot_roc(df, label, save_path):
    years = [1, 2, 3, 4, 5]
    auc_values = []
    for time in years:
        print(len(df))
        df = df[~((df['fu_days'] < time * 365) & (df[label] == 0))]
        labels = (df['fu_days'] >= time * 365).astype(int)

        auc = roc_auc_score(labels, df[f'risk_scores_{time}'])
        auc_values.append(auc)
    print('auc: ', auc_values)
    plt.plot(years, auc_values, marker='o', linestyle='-', color='b')

    for i in range(len(years)):
        plt.text(years[i], auc_values[i], f'{auc_values[i]:.3f}', fontsize=12, ha='center', color='red')

    plt.title('AUC for Survival Network (1 to 5 Years)')
    plt.xlabel('Years')
    plt.ylabel('AUC')
    plt.ylim(0, 1)
    plt.xticks(years)
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def plot_cindex(df, label, save_path):
    years = [1, 2, 3, 4, 5]
    c_indexs = []
    df = df[df[label] == 1]
    for time in years:
        c_index = concordance_index(df['fu_days'], df[f'risk_scores_{time}'], df[label])
        c_indexs.append(c_index)

    plt.plot(years, c_indexs, marker='o', linestyle='-', color='b')

    for i in range(len(years)):
        plt.text(years[i], c_indexs[i], f'{c_indexs[i]:.3f}', fontsize=12, ha='center', color='red')

    # 设置图形的标题和标签
    plt.title('C-Index for Survival Network (1 to 5 Years)')
    plt.xlabel('Years')
    plt.ylabel('C-Index')
    plt.ylim(0, 1)
    plt.xticks(years)
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def main():
    model = ResNet34(num_classes=n_intervals)
    h5_path = r'data/ecg_data.h5'
    h5_csv_path = r'data/ecg_labels.csv'

    df_out = eval_batch(model, 'dataset/VHD/tset_VHD.csv', 'fu_label', 'checkpoints/', h5_path, h5_csv_path)

    plot_roc(df_out, 'fu_lvef_label', 'results/ROC.png')
    plot_cindex(df_out, 'fu_lvef_label', 'results/c-index.png')


if __name__ == "__main__":
    device_ids = [0]
    breaks = np.arange(0., 365. * 5, 365. / 8)
    n_intervals = len(breaks) - 1
    main()
