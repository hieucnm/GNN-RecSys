import gc
from abc import ABC
import torch
from dgl import heterograph
from src.utils_data import create_common_ids, read_data_rename_id, read_data
import numpy as np
import functools


class BaseDataSet:
    def __init__(self, train_iid_map_df=None, use_edge_features=False):

        # basic attributes for this dataset
        self.user_id = 'src_id'
        self.item_id = 'ad_cate'
        self.des_uid = 'des_id'
        self.user_ids = [self.user_id, self.des_uid]
        self.suffix = 'idx'  # e.g: `src_id` will be map to `src_id_idx`
        self.user_idx = f'{self.user_id}_{self.suffix}'
        self.des_uidx = f'{self.des_uid}_{self.suffix}'
        self.item_idx = f'{self.item_id}_{self.suffix}'
        self.ad_kind = ['impression', 'click', 'conv']
        self.ex_cols = [self.user_id, self.item_id, self.des_uid,
                        self.user_idx, self.item_idx, self.des_uidx, 'kind', 'group_id']

        self.uid_map_df = None
        self.iid_map_df = None
        self.graph = None
        self.train_graph = None
        self.use_edge_features = use_edge_features
        if train_iid_map_df is not None:
            train_iid_map_df[self.item_id] = train_iid_map_df[self.item_id].astype(np.int32)
            train_iid_map_df[self.item_idx] = train_iid_map_df[self.item_idx].astype(np.int32)
            self.iid_map_df = train_iid_map_df

    @property
    def _ad_message_edges(self):
        return {
            'click': [(self.user_id, 'clicked', self.item_id), (self.item_id, 'clicked-by', self.user_id)],
            'conv': [(self.user_id, 'converted', self.item_id), (self.item_id, 'converted-by', self.user_id)],
            'impression': [(self.user_id, 'impressed', self.item_id), (self.item_id, 'impressed-by', self.user_id)],
        }

    @property
    def _message_edges(self):
        return self._ad_message_edges

    def _supervision_edges(self):
        return {'label_1': self.pos_label_edge_types, 'label_0': self.neg_label_edge_types}

    @property
    def label_edge_types(self):
        return self.pos_label_edge_types + self.neg_label_edge_types

    @property
    def pos_label_edge_types(self):
        return [(self.user_id, 'will-convert', self.item_id)]

    @property
    def neg_label_edge_types(self):
        return [(self.user_id, 'will-click', self.item_id)]

    @property
    def iid_columns(self):
        return self.iid_map_df[self.item_id].tolist()

    @property
    def num_users(self):
        return self.uid_map_df.shape[0]

    @property
    def num_items(self):
        return self.iid_map_df.shape[0]

    @property
    def num_user_features(self):
        return self.train_graph.srcdata['features'][self.user_id].shape[1]

    @property
    def num_nodes_dict(self):
        return {self.user_id: self.num_users, self.item_id: self.num_items}

    @property
    def num_edge_features_dict(self):
        assert self.use_edge_features, "Cannot get `num_edge_features_dict` because `use_edge_features` is False"
        assert self.train_graph, "Your graph wasn't initialized!"
        return {
            e_type: feats.shape[1]
            for e_type, feats in self.train_graph.edata['features'].items()
        }

    @property
    @functools.lru_cache()
    def item2node(self):
        return self.iid_map_df.set_index(self.item_id)[self.item_idx].to_dict()

    @property
    @functools.lru_cache()
    def node2item(self):
        return self.iid_map_df.set_index(self.item_idx)[self.item_id].to_dict()

    @property
    def model_edge_types(self):
        assert self.train_graph, "Your graph wasn't initialized!"
        return self.train_graph.canonical_etypes

    def _index_item_id(self, df_list):
        # df_ad, df_label
        if self.iid_map_df is not None:
            df_list_res = []
            for df in df_list:
                df_list_res.append(df.merge(self.iid_map_df, on=self.item_id))
        else:
            df_list_res, self.iid_map_df = create_common_ids(df_list, [self.item_id], self.suffix)
        return df_list_res

    def _index_user_id(self, user_profile, df_list):
        self.uid_map_df = user_profile[[self.user_id]]
        self.uid_map_df[self.user_idx] = self.uid_map_df.index.astype(np.int32)
        df_list_res = []
        for df in df_list:
            if self.des_uid in df.columns:
                df_list_res.append(
                    df.merge(self.uid_map_df, on=self.user_id).merge(
                        self.uid_map_df.rename(columns={self.user_id: self.des_uid, self.user_idx: self.des_uidx}),
                        on=self.des_uid)
                )
            else:
                df_list_res.append(df.merge(self.uid_map_df, on=self.user_id))
        return df_list_res

    def _init_graph_schema(self, df_ad):
        graph_schema = dict()
        for kind in self.ad_kind:
            e_type, reverse_e_type = self._message_edges[kind]
            pairs = df_ad[df_ad['kind'] == kind][[self.user_idx, self.item_idx]].values
            graph_schema[e_type] = (pairs[:, 0], pairs[:, 1])
            graph_schema[reverse_e_type] = (pairs[:, 1], pairs[:, 0])
        return graph_schema

    def _import_features(self, user_profile, df_ad):
        self.train_graph.nodes[self.user_id].data['features'] = torch.FloatTensor(user_profile.drop(columns=[self.user_id]).values)
        self.train_graph.nodes[self.item_id].data['features'] = torch.IntTensor(list(range(self.iid_map_df.shape[0])))
        for kind in self.ad_kind:
            e_type, reverse_e_type = self._message_edges[kind]
            ad_features = torch.FloatTensor(df_ad[df_ad['kind'] == kind][df_ad.columns.difference(self.ex_cols)].values)
            self.train_graph.edges[e_type].data['features'] = ad_features
            self.train_graph.edges[reverse_e_type].data['features'] = ad_features

    def _init_supervision_graph(self, df_label):
        supervision_graph_schema = {}
        for lab, e_type in zip([0, 1], self.label_edge_types):
            pairs = df_label[df_label['label'] == lab][[self.user_idx, self.item_idx]].values
            supervision_graph_schema[e_type] = (pairs[:, 0], pairs[:, 1])
        self.graph = heterograph(supervision_graph_schema, num_nodes_dict=self.num_nodes_dict, idtype=torch.int32)

    def init_graph(self):
        """ Initialize graph from data directly"""

        user_profile, df_ad, df_label = self._load_data()

        # Index item_id (e.g: `ad_cate_idx`), use indices from training data or create new ones
        df_ad, df_label = self._index_item_id([df_ad, df_label])
        gc.collect()

        # Index user_id (e.g: `src_id_idx`)
        df_ad, df_label = self._index_user_id(user_profile, [df_ad, df_label])
        gc.collect()

        # Create graph
        graph_schema = self._init_graph_schema(df_ad)
        self.train_graph = heterograph(graph_schema, num_nodes_dict=self.num_nodes_dict, idtype=torch.int32)

        # Import features
        self._import_features(user_profile, df_ad)

        # Create supervision graph
        self._init_supervision_graph(df_label)
        print(self.summary_graph())

    def summary_graph(self):
        return f"Summary graph : train_graph: {self.train_graph}"

    def _load_data(self):
        # TODO: return `user_profile`, `df_ad` and `df_label`
        raise NotImplementedError


class TrainBaseDataSet(BaseDataSet):
    def __init__(self, data_dirs, train_iid_map_df=None, use_edge_features=False):
        super(TrainBaseDataSet, self).__init__(train_iid_map_df=train_iid_map_df, use_edge_features=use_edge_features)
        self.data_dirs = [x.rstrip('/') for x in data_dirs.split(',')]

    def _load_data(self):
        if len(self.data_dirs) > 1:
            df_ad = read_data_rename_id(self.data_dirs, 'ad.parquet', self.user_ids)
            user_profile = read_data_rename_id(self.data_dirs, 'user_features.parquet', self.user_ids)
            df_label = read_data_rename_id(self.data_dirs, 'label.parquet', self.user_ids)
        else:
            # if only 1 data_dir, don't rename user_id
            data_dir = self.data_dirs[0]
            df_ad = read_data(data_dir + '/ad.parquet')
            user_profile = read_data(data_dir + '/user_features.parquet')
            df_label = read_data(data_dir + '/label.parquet')
        return user_profile, df_ad, df_label


class GroupChatDataSet(BaseDataSet, ABC):

    @property
    def _message_edges(self):
        msg_edge = super()._message_edges
        msg_edge.update({'group_chat': [(self.user_id, 'group-chat-to', self.user_id)]})
        return msg_edge

    def init_graph(self):
        """ Initialize graph from data directly"""
        user_profile, df_ad, df_label, df_group = self._load_data()

        # Index item_id (e.g: `ad_cate_idx`), use indices from training data or create new ones
        df_ad, df_label = self._index_item_id([df_ad, df_label])
        gc.collect()

        # Index user_id (e.g: `src_id_idx`)
        df_ad, df_label, df_group = self._index_user_id(user_profile, [df_ad, df_label, df_group])
        gc.collect()

        # Create graph (first, init ad edges using parent class, # then, add group edges)
        graph_schema = super()._init_graph_schema(df_ad)
        graph_schema[self._message_edges['group_chat'][0]] = (df_group[self.user_idx].values, df_group[self.des_uidx].values)
        self.train_graph = heterograph(graph_schema, num_nodes_dict=self.num_nodes_dict, idtype=torch.int32)

        # Import features
        self._import_features(user_profile, df_ad)
        self.train_graph.edges[self._message_edges['group_chat'][0]].data['features'] = \
            torch.FloatTensor(df_group[df_group.columns.difference(self.ex_cols)].values)

        # Create supervision graph
        self._init_supervision_graph(df_label)
        print(self.summary_graph())


class TrainGroupChatDataSet(GroupChatDataSet, TrainBaseDataSet):
    def _load_data(self):
        user_profile, df_ad, df_label = super()._load_data()
        if len(self.data_dirs) > 1:
            df_group = read_data_rename_id(self.data_dirs, 'group_chat.parquet', self.user_ids)
        else:
            df_group = read_data(self.data_dirs[0] + '/group_chat.parquet')
        return user_profile, df_ad, df_label, df_group


# ===================================================
# ===================================================


class ObjectView(object):
    def __init__(self, d):
        self.__dict__ = d
