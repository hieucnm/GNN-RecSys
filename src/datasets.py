import pandas as pd
import torch
from dgl import heterograph

from src.utils_data import create_common_ids, read_data_change_uid


class DataSet:
    def __init__(self, data_dirs, train_iid_map_df=None, has_label=True):
        self.new_id_suffix = 'idx'  # e.g: `src_id` will be map to `src_id_idx`
        self.homo_data_names = ['group_chat']  # these data contains user - user interactions
        self.label_edge_types = self._edge_triplets['label']
        self.train_iid_map_df = train_iid_map_df
        self.has_label = has_label

        self.data_dirs = data_dirs.split(',')
        self.data_dict, self.uid_map_df, self.iid_map_df = self.init_data()
        self.feature_dict = self._load_features()
        self.summary_data()

    @property
    def user_id(self):
        return 'src_id'

    @property
    def item_id(self):
        return 'ad_cate'

    @property
    def _edge_triplets(self):
        return {
            'group_chat': [('src_id', 'group-chat-to', 'src_id'), ('src_id', 'group-chat-by', 'src_id')],
            'ad_click': [('src_id', 'clicked', 'ad_cate'), ('ad_cate', 'clicked-by', 'src_id')],
            'ad_convert': [('src_id', 'converted', 'ad_cate'), ('ad_cate', 'converted-by', 'src_id')],
            'label': [('src_id', 'will-convert', 'ad_cate')],
        }

    @property
    def num_items(self):
        return self.iid_map_df.shape[0]

    @property
    def num_users(self):
        return self.uid_map_df.shape[0]

    @property
    def num_user_features(self):
        return self.feature_dict[self.user_id].shape[1]

    def get_node2uid_dict(self):
        return self.uid_map_df.set_index(f'{self.user_id}_{self.new_id_suffix}')[self.user_id].to_dict()

    def get_node2iid_dict(self):
        return self.iid_map_df.set_index(f'{self.item_id}_{self.new_id_suffix}')[self.item_id].to_dict()

    def init_data(self):
        # oa_form: both click and convert
        df_ad = pd.concat([read_data_change_uid(data_dir.rstrip('/') + '/ad.parquet', index)
                           for index, data_dir in enumerate(self.data_dirs)]).reset_index(drop=True)
        # group chat
        df_group = pd.concat([read_data_change_uid(data_dir.rstrip('/') + '/group_chat.parquet', index)
                              for index, data_dir in enumerate(self.data_dirs)]).reset_index(drop=True)

        # oa_form in future, convert only
        if self.has_label:
            df_label = pd.concat([read_data_change_uid(data_dir.rstrip('/') + '/label.parquet', index)
                                  for index, data_dir in enumerate(self.data_dirs)]).reset_index(drop=True)
        else:
            # empty dataframe
            df_label = pd.DataFrame(columns=[self.user_id, self.item_id])

        # map src_id and des_id of different dataframes to same indices
        (df_group, df_ad, df_label), uid_map_df = create_common_ids([df_group, df_ad, df_label],
                                                                    ['src_id', 'des_id'],
                                                                    self.new_id_suffix)
        # map ad_cate of different dataframes to same indices
        (df_ad, df_label), iid_map_df = create_common_ids([df_ad, df_label],
                                                          ['ad_cate'],
                                                          self.new_id_suffix)

        # Ensure index of items in this dataset is the same as that in the training dataset
        if self.train_iid_map_df is not None:
            iid_map_df = self.train_iid_map_df
            df_ad = self._use_train_iid_map(df_ad)
            df_label = self._use_train_iid_map(df_label)

        data_dict = {
            'group_chat': df_group,
            'ad_click': df_ad[df_ad['kind'] == 100].reset_index(drop=True),
            'ad_convert': df_ad[df_ad['kind'] == 201].reset_index(drop=True),
            'label': df_label
        }
        return data_dict, uid_map_df, iid_map_df

    def _use_train_iid_map(self, df):
        df = df.drop(columns=[f'{self.item_id}_{self.new_id_suffix}'])
        df = df.merge(self.train_iid_map_df, on=self.item_id)
        return df

    def _load_features(self):
        feature_dict = {}

        # user feature
        user_feature = pd.concat([read_data_change_uid(data_dir.rstrip('/') + '/user_features.parquet', index)
                                  for index, data_dir in enumerate(self.data_dirs)]).reset_index(drop=True)
        user_feature = user_feature.merge(self.uid_map_df, on=self.user_id)
        assert user_feature.shape[0] == self.uid_map_df.shape[0], \
            "no. users in user_feat ({}) not equal no. users in interactions ({})".format(user_feature.shape[0],
                                                                                          self.uid_map_df.shape[0])
        user_feature = user_feature.iloc[:, 1:]  # the first column always is user_id
        feature_dict[self.user_id] = torch.tensor(user_feature.values).float()

        # item feature: index only
        item_index = list(range(self.num_items))
        feature_dict[self.item_id] = torch.tensor(item_index).long()

        # also load other node's features here if necessary
        return feature_dict

    def summary_data(self):
        df_group = self.data_dict['group_chat']
        df_click = self.data_dict['ad_click']
        df_convert = self.data_dict['ad_convert']
        df_label = self.data_dict['label']

        summary = "===== Data Summary =====\n" \
                  "- group_chat: #rows = {:8d}\n" \
                  "- ad_click  : #rows = {:8d} | #users = {:8d} | #items = {:2d}\n" \
                  "- ad_convert: #rows = {:8d} | #users = {:8d} | #items = {:2d}\n" \
                  "- label     : #rows = {:8d} | #users = {:8d} | #items = {:2d}\n" \
                  "- Final     :         {:8s} | #users = {:8d} | #items = {:2d} | #user_features = {:3d}".format(
            df_group.shape[0],
            df_click.shape[0], df_click[self.user_id].nunique(), df_click[self.item_id].nunique(),
            df_convert.shape[0], df_convert[self.user_id].nunique(), df_convert[self.item_id].nunique(),
            df_label.shape[0], df_label[self.user_id].nunique(), df_label[self.item_id].nunique(),
            '', self.uid_map_df.shape[0], self.iid_map_df.shape[0], self.feature_dict[self.user_id].shape[1]
        )
        print(summary)

    def init_graph_schema(self):
        graph_schema = dict()
        for data_name, df in self.data_dict.items():
            if not self.has_label and data_name == 'label':
                continue
            if data_name in self.homo_data_names:
                edge_type = self._edge_triplets[data_name][0]
                reverse_edge_type = self._edge_triplets[data_name][1]
                src_node = f'src_id_{self.new_id_suffix}'
                dst_node = f'des_id_{self.new_id_suffix}'
                graph_schema[edge_type] = (df[src_node].values, df[dst_node].values)
                graph_schema[reverse_edge_type] = (df[dst_node].values, df[src_node].values)
            else:
                for edge_type in self._edge_triplets[data_name]:
                    src_node, _, dst_node = edge_type
                    src_node = f'{src_node}_{self.new_id_suffix}'
                    dst_node = f'{dst_node}_{self.new_id_suffix}'
                    graph_schema[edge_type] = (df[src_node].values, df[dst_node].values)
        return graph_schema

    def _import_feature(self, graph):
        graph.nodes[self.user_id].data['features'] = self.feature_dict[self.user_id]
        graph.nodes[self.item_id].data['features'] = self.feature_dict[self.item_id]
        return graph

    def init_graph(self):
        graph_schema = self.init_graph_schema()
        graph = heterograph(graph_schema)
        graph = self._import_feature(graph)
        return graph


class ObjectView(object):
    def __init__(self, d):
        self.__dict__ = d
