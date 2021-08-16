import os
import numpy as np
import pandas as pd
import pickle
import datetime as dt
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


def save_everything(trained_model, graph, data, params, fixed_params):
    timestamp = str(dt.datetime.now()).replace(' ', '')
    torch.save(trained_model.state_dict(), f'models/{timestamp}.pth')
    # Save all necessary params
    save_outputs(
        {
            f'{timestamp}_params': params,
            f'{timestamp}_fixed_params': vars(fixed_params),
        },
        'models/'
    )
    print("Saved model & parameters to disk.")

    # Save graph & ID mapping
    save_graphs(f'models/{timestamp}_graph.bin', [graph])
    save_outputs(
        {
            f'{timestamp}_user_id': data.user_id_df,
            f'{timestamp}_item_id': data.item_id_df,
        },
        'models/'
    )
    print("Saved graph & ID mapping to disk.")
