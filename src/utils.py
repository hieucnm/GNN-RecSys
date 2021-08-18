import os
import numpy as np
import pandas as pd
import pickle
import json
import torch
from dgl.data.utils import save_graphs


def save_txt(data_to_save, filepath, mode='a'):
    """
    Save text to a file.
    """
    with open(filepath, mode) as text_file:
        text_file.write(data_to_save + '\n')


def save_outputs(files_to_save: dict,
                 folder_path):
    """
    Save objects as pickle files, in a given folder.
    """
    for name, file in files_to_save.items():
        with open(folder_path + name + '.pkl', 'wb') as f:
            pickle.dump(file, f)


def get_last_checkpoint():
    """
    Fetch path of last checkpoint available in the root folder, based on the date in the filename.
    """
    logdir = '.'
    logfiles = sorted([f for f in os.listdir(logdir) if f.startswith('checkpoint')])
    checkpoint_path = logfiles[-1]
    return checkpoint_path


def read_data(file_path):
    """
    Generic function to read any kind of data. Extensions supported: '.gz', '.csv', '.pkl'
    """
    if isinstance(file_path, pd.DataFrame):
        return file_path

    if file_path.endswith('.gz'):
        obj = pd.read_csv(file_path, compression='gzip',
                          header=0, sep=';', quotechar='"',
                          error_bad_lines=False)
    elif file_path.endswith('.csv'):
        obj = pd.read_csv(file_path)
    elif file_path.endswith('.parquet'):
        obj = pd.read_parquet(file_path)
    elif file_path.endswith('.pkl'):
        with open(file_path, 'rb') as handle:
            obj = pickle.load(handle)
    else:
        raise KeyError('File extension of {} not recognized.'.format(file_path))
    return obj


def softmax(x):
    """
    (Currently not used.) Compute softmax values for each sets of scores in x.
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def save_everything(trained_model, graph, data, params, fixed_params, save_dir):

    torch.save(trained_model.state_dict(), f'{save_dir}/model.pth')
    print("Model saved!")

    # Save all necessary params
    pickle.dump(params, open(f'{save_dir}/params.pkl', 'wb'))
    pickle.dump(fixed_params, open(f'{save_dir}/fixed_params.pkl', 'wb'))
    print("Params and fixed params saved!")

    # Save graph & ID mapping
    save_graphs(f'{save_dir}/graph.bin', [graph])
    data.user_id_df.to_csv(f'{save_dir}/user_id.csv', index=False)
    data.item_id_df.to_csv(f'{save_dir}/item_id.csv', index=False)
    print("Graph & ID mapping saved!")
    print(f"Finish saving everything at {save_dir}")
