import json

import matplotlib.pyplot as plt
from datetime import datetime
import textwrap
from os import makedirs

import numpy as np


def plot_train_loss(hp_sentence, viz, save_dir):
    """
    Visualize train & validation loss & metrics. hp_sentence is used as the title of the plot.

    Saves plots in the plots folder.
    """
    json.dump(viz, open(f"{save_dir}/viz_data.json", "w"))

    if 'val_loss_list' in viz.keys():
        fig = plt.figure()
        x = np.arange(len(viz['train_loss_list']))
        plt.title('Losses')
        fig.tight_layout()
        plt.rcParams["axes.titlesize"] = 6
        plt.plot(x, viz['train_loss_list'])
        plt.plot(x, viz['val_loss_list'])
        plt.legend(['training loss', 'valid loss'], loc='upper left')
        plt.savefig(f'{save_dir}/loss.png')
        plt.close(fig)

    if 'val_recall_list' in viz.keys():
        fig = plt.figure()
        x = np.arange(len(viz['train_precision_list']))
        plt.title('Metrics')
        fig.tight_layout()
        plt.rcParams["axes.titlesize"] = 6
        plt.plot(x, viz['train_precision_list'])
        plt.plot(x, viz['train_recall_list'])
        plt.plot(x, viz['train_coverage_list'])
        plt.plot(x, viz['train_auc_list'])
        plt.plot(x, viz['val_precision_list'])
        plt.plot(x, viz['val_recall_list'])
        plt.plot(x, viz['val_coverage_list'])
        plt.plot(x, viz['val_auc_list'])
        plt.legend(['train precision', 'train recall', 'train coverage/10', 'train auc',
                    'valid precision', 'valid recall', 'valid coverage/10', 'valid_auc'], loc='upper left')
        plt.savefig(f'{save_dir}/metrics.png')
        plt.close(fig)
