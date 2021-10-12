import json
import os
import pickle

import numpy as np
import pandas as pd


# ====================
# PREPROCESS STAGE ===

def preprocess_features(df, bool_features=None, numeric_features=None, mean_dict={}):
    if bool_features is not None:
        for feat in bool_features:
            df[f'{feat}_0'] = (df[feat] == 0).astype(np.int32)
            df[f'{feat}_1'] = (df[feat] == 1).astype(np.int32)
            df[f'{feat}_nan'] = df[feat].isnull().astype(np.int32)
            df = df.drop(columns=[feat])

    if numeric_features is not None:
        for feat in numeric_features:
            if feat in mean_dict:
                df[f"{feat}_nan"] = df[feat].isnull().astype(np.int32)
                df[feat] = df[feat].fillna(mean_dict[feat]).astype(np.float32)

    return df


def preprocess_device(df, device_brand_list, device_col="chatmobile_most_used_device"):
    def get_device_brand(x):
        try:
            brand = x.split()[0]
            if brand.startswith('ip') or brand == 'apple':
                return 'ip'
            if brand in device_brand_list:
                return brand
            return 'other'
        except:
            return 'nan'

    df[device_col] = df[device_col].apply(get_device_brand)
    for brand in device_brand_list + ['nan', 'other']:
        df[f'device_{brand}'] = (df[device_col] == brand).astype(np.int32)
    df = df.drop(columns=[device_col])
    return df


def preprocess_user_profiles(df, metadata):
    df = preprocess_device(df)
    df = preprocess_features(df,
                             bool_features=metadata['USER_BOOL_FEATURES'],
                             numeric_features=metadata['USER_NUMERIC_FEATURES'],
                             mean_dict=metadata['USER_MEAN_DICT'])

    # no scaling, we already had batch_norm layer
    return df


def preprocess_group_features(df, metadata):
    df = df.drop(columns=['src_last_chat_time', 'des_last_chat_time', 'last_active_time'])
    df = preprocess_features(df,
                             bool_features=metadata['GROUP_BOOL_FEATURES'],
                             numeric_features=metadata['GROUP_NUMERIC_FEATURES'],
                             mean_dict=metadata['GROUP_MEAN_DICT'])

    # scale
    scaler_params = metadata['GROUP_SCALER_PARAMS']
    df[scaler_params['features']] = (df[scaler_params['features']] - scaler_params['mean']) / scaler_params['std']
    return df


def preprocess_ad_features(df, metadata):
    # All ad features are non-null, so we dont need to preprocess, just scale
    df = df.drop(columns=['last_active_time'])
    scaler_params = metadata['AD_SCALER_PARAMS']
    df[scaler_params['features']] = (df[scaler_params['features']] - scaler_params['mean']) / scaler_params['std']
    return df


# =====================
# BUILD GRAPH STAGE ===

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
    elif file_path.endswith('.parquet') or os.path.isdir(file_path):
        obj = pd.read_parquet(file_path)
    elif file_path.endswith('json'):
        obj = json.load(open(file_path))
    elif file_path.endswith('pkl'):
        obj = pickle.load(open(file_path, 'rb'))
    else:
        raise KeyError('File extension of {} not recognized.'.format(file_path))
    return obj


user_id = 'src_id'
item_id = 'ad_cate'
des_uid = 'des_id'
user_ids = [user_id, des_uid]
suffix = 'idx'
user_idx = f'{user_id}_{suffix}'
des_uidx = f'{des_uid}_{suffix}'
item_idx = f'{item_id}_{suffix}'
ad_kind = ['impression', 'click', 'conv']
non_feature_columns = [user_id, item_id, des_uid, user_idx, item_idx, des_uidx, 'kind', 'group_id']

ad_message_edges = {
    'click': [(user_id, 'clicked', item_id), (item_id, 'clicked-by', user_id)],
    'conv': [(user_id, 'converted', item_id), (item_id, 'converted-by', user_id)],
    'impression': [(user_id, 'impressed', item_id), (item_id, 'impressed-by', user_id)],
}

message_edges = ad_message_edges
message_edges.update({'group_chat': [(user_id, 'group-chat-to', user_id)]})
