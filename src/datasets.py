# import pandas as pd
# import torch
# from dgl import heterograph
# from src.utils_data import create_common_ids, read_data_from_multiple_dirs
#
#
# class BaseDataSet:
#     def __init__(self, has_label=True, train_iid_map_df=None):
#         self.new_id_suffix = 'idx'  # e.g: `src_id` will be map to `src_id_idx`
#         self.homo_data_names = ['group_chat']  # these data contains user - user interactions
#         self.user_id = 'src_id'
#         self.item_id = 'ad_cate'
#         self.user_ids = ['src_id', 'des_id']
#         self.feature_dict = None
#         self.data_dict = None
#         self.uid_map_df = None
#         self.iid_map_df = None
#         self.has_label = has_label
#         self.train_iid_map_df = train_iid_map_df
#
#     @property
#     def _edge_triplets(self):
#         triplets_dict = {
#             'group_chat': [('src_id', 'group-chat-to', 'src_id'), ('src_id', 'group-chat-by', 'src_id')],
#             'ad_click': [('src_id', 'clicked', 'ad_cate'), ('ad_cate', 'clicked-by', 'src_id')],
#             'ad_convert': [('src_id', 'converted', 'ad_cate'), ('ad_cate', 'converted-by', 'src_id')],
#         }
#         if self.has_label:
#             triplets_dict.update({'label': self.label_edge_types})
#         return triplets_dict
#
#     @property
#     def label_edge_types(self):
#         if self.has_label:
#             return [('src_id', 'will-convert', 'ad_cate')]
#         return None
#
#     @property
#     def num_items(self):
#         return self.iid_map_df.shape[0]
#
#     @property
#     def num_users(self):
#         return self.uid_map_df.shape[0]
#
#     def get_node2iid_dict(self):
#         return self.iid_map_df.set_index(f'{self.item_id}_{self.new_id_suffix}')[self.item_id].to_dict()
#
#     def get_node2uid_dict(self):
#         return self.uid_map_df.set_index(f'{self.user_id}_{self.new_id_suffix}')[self.user_id].to_dict()
#
#     def get_user2nid_dict(self):
#         return self.uid_map_df.set_index(self.user_id)[f'{self.user_id}_{self.new_id_suffix}'].to_dict()
#
#     def get_item_id_columns(self):
#         return self.iid_map_df[self.item_id].tolist()
#
#     @property
#     def num_user_features(self):
#         return self.feature_dict[self.user_id].shape[1]
#
#     def summary_data(self):
#         df_group = self.data_dict['group_chat']
#         df_click = self.data_dict['ad_click']
#         df_convert = self.data_dict['ad_convert']
#         if 'label' in self.data_dict:
#             df_label = self.data_dict['label']
#         else:
#             df_label = pd.DataFrame(columns=[self.user_id, self.item_id])
#         summary = "===== Data Summary =====\n" \
#                   "- group_chat: #rows = {:8d}\n" \
#                   "- ad_click  : #rows = {:8d} | #users = {:8d} | #items = {:3d}\n" \
#                   "- ad_convert: #rows = {:8d} | #users = {:8d} | #items = {:3d}\n" \
#                   "- label     : #rows = {:8d} | #users = {:8d} | #items = {:3d}\n" \
#                   "- Union     :         {:8s} | #users = {:8d} | #items = {:3d}\n" \
#                   "- user_feats:         {:8s} | #users = {:8d} | #feats = {:3d}".format(
#             df_group.shape[0],
#             df_click.shape[0], df_click[self.user_id].nunique(), df_click[self.item_id].nunique(),
#             df_convert.shape[0], df_convert[self.user_id].nunique(), df_convert[self.item_id].nunique(),
#             df_label.shape[0], df_label[self.user_id].nunique(), df_label[self.item_id].nunique(),
#             '', self.uid_map_df.shape[0], self.iid_map_df.shape[0],
#             '', self.feature_dict[self.user_id].shape[0], self.feature_dict[self.user_id].shape[1] - 1
#         )
#         print(summary)
#
#     def init_graph_schema(self):
#         graph_schema = dict()
#         for data_name, df in self.data_dict.items():
#             if data_name in self.homo_data_names:
#                 edge_type = self._edge_triplets[data_name][0]
#                 reverse_edge_type = self._edge_triplets[data_name][1]
#                 src_node = f'src_id_{self.new_id_suffix}'
#                 dst_node = f'des_id_{self.new_id_suffix}'
#                 graph_schema[edge_type] = (df[src_node].values, df[dst_node].values)
#                 graph_schema[reverse_edge_type] = (df[dst_node].values, df[src_node].values)
#             else:
#                 for edge_type in self._edge_triplets[data_name]:
#                     src_node, _, dst_node = edge_type
#                     src_node = f'{src_node}_{self.new_id_suffix}'
#                     dst_node = f'{dst_node}_{self.new_id_suffix}'
#                     graph_schema[edge_type] = (df[src_node].values, df[dst_node].values)
#         return graph_schema
#
#     def _import_feature(self, g):
#         g.nodes[self.user_id].data['features'] = self.feature_dict[self.user_id]
#         g.nodes[self.item_id].data['features'] = self.feature_dict[self.item_id]
#         return g
#
#     def init_graph(self):
#         graph_schema = self.init_graph_schema()
#         num_nodes_dict = {
#             self.user_id: self.uid_map_df.shape[0],
#             self.item_id: self.iid_map_df.shape[0]
#         }
#         graph = heterograph(graph_schema, num_nodes_dict=num_nodes_dict)
#         graph = self._import_feature(graph)
#         return graph
#
#     def _use_train_iid_map(self, df_list, train_iid_map_df):
#         for df in df_list:
#             df = df.drop(columns=[f'{self.item_id}_{self.new_id_suffix}'])
#             df = df.merge(train_iid_map_df, on=self.item_id)
#             yield df
#
#     def _verify_all_user_feature_exist(self, user_feature, user_data_list):
#         id_set = set()
#         _ = [id_set.update(df[id_column]) for df in user_data_list
#              for id_column in self.user_ids if id_column in df.columns]
#         diff_users = id_set.difference(user_feature[self.user_id])
#         assert len(diff_users) == 0, f"There are {len(diff_users)} having no features"
#
#     def load_data(self, print_summary=True):
#
#         user_feature, df_ad, df_group, df_label = self._load_data()
#         if df_label is None:
#             # dummy data, will not be added to data_dict later
#             df_label = pd.DataFrame(columns=[self.user_id, self.item_id])
#
#         self._verify_all_user_feature_exist(user_feature=user_feature,
#                                             user_data_list=[df_ad, df_group, df_label]
#                                             )
#
#         # map src_id and des_id of different dataframes to same indices
#         (df_group, df_ad, df_label, user_feature), uid_map_df = create_common_ids(
#             [df_group, df_ad, df_label, user_feature], self.user_ids, self.new_id_suffix
#         )
#
#         # map ad_cate of different dataframes to same indices
#         (df_ad, df_label), iid_map_df = create_common_ids([df_ad, df_label],
#                                                           [self.item_id],
#                                                           self.new_id_suffix)
#
#         # Ensure index of items in this dataset is the same as that in the training dataset
#         if self.train_iid_map_df is not None:
#             iid_map_df = self.train_iid_map_df
#             df_ad, df_label = self._use_train_iid_map([df_ad, df_label], self.train_iid_map_df)
#
#         data_dict = {
#             'group_chat': df_group,
#             'ad_click': df_ad[df_ad['kind'] == 100].reset_index(drop=True),
#             'ad_convert': df_ad[df_ad['kind'] == 201].reset_index(drop=True)
#         }
#         if df_label.shape[0] != 0:
#             data_dict['label'] = df_label
#
#         self.data_dict = data_dict
#         self.uid_map_df = uid_map_df
#         self.iid_map_df = iid_map_df
#
#         user_feature = user_feature.sort_values(by=f'{self.user_id}_{self.new_id_suffix}', ascending=True) \
#             .drop(columns=[self.user_id, f'{self.user_id}_{self.new_id_suffix}'])
#         user_feature = torch.tensor(user_feature.values).float()
#
#         item_feature = torch.tensor(list(range(iid_map_df.shape[0]))).long()
#
#         self.feature_dict = {
#             self.user_id: user_feature,
#             self.item_id: item_feature
#         }
#
#         if print_summary:
#             self.summary_data()
#
#     def _load_data(self):
#         raise NotImplementedError
#
#
# class TrainDataSet(BaseDataSet):
#     def __init__(self, data_dirs, train_iid_map_df=None):
#         super(TrainDataSet, self).__init__(has_label=True, train_iid_map_df=train_iid_map_df)
#         self.train_iid_map_df = train_iid_map_df
#         self.data_dirs = [x.rstrip('/') for x in data_dirs.split(',')]
#
#     def _load_data(self):
#         df_ad = read_data_from_multiple_dirs(dir_list=self.data_dirs, filename='ad.parquet')
#         df_group = read_data_from_multiple_dirs(dir_list=self.data_dirs, filename='group_chat.parquet')
#         user_feature = read_data_from_multiple_dirs(dir_list=self.data_dirs, filename='user_features.parquet')
#         df_label = read_data_from_multiple_dirs(dir_list=self.data_dirs, filename='label.parquet')
#         return user_feature, df_ad, df_group, df_label
#
#
# class InferenceDataSet(BaseDataSet):
#     def __init__(self, train_iid_map_df, df_group, df_ad, user_feature, to_infer_uid_df=None):
#         super(InferenceDataSet, self).__init__(has_label=False)
#         self.train_iid_map_df = train_iid_map_df
#         self.user_feature = user_feature
#         self.df_group = df_group
#         self.df_ad = df_ad
#
#         if to_infer_uid_df is not None:
#             self.to_infer_uid_df = to_infer_uid_df
#             self._verify_all_user_feature_exist(user_feature, [to_infer_uid_df])
#         else:
#             self.to_infer_uid_df = user_feature[[self.user_id]]
#
#     def _load_data(self):
#         return self.user_feature, self.df_ad, self.df_group, None
#
#     def get_infer_user_node_id(self):
#         user2nid = self.get_user2nid_dict()
#         to_infer_user_id = self.to_infer_uid_df[self.user_id].tolist()
#         to_infer_node_id = [user2nid[uid] for uid in to_infer_user_id]
#         return to_infer_node_id
#
#
# class ObjectView(object):
#     def __init__(self, d):
#         self.__dict__ = d


import pandas as pd
import torch
from dgl import heterograph
from src.utils_data import create_common_ids, read_data_from_multiple_dirs, read_data


class BaseDataSet:
    def __init__(self, has_label=True, train_iid_map_df=None):
        self.new_id_suffix = 'idx'  # e.g: `src_id` will be map to `src_id_idx`
        self.homo_data_names = ['group_chat']  # these data contains user - user interactions
        self.user_id = 'src_id'
        self.item_id = 'ad_cate'
        self.user_ids = ['src_id', 'des_id']
        self.feature_dict = None
        self.data_dict = None
        self.uid_map_df = None
        self.iid_map_df = None
        self.graph = None
        self.adjust_graph = None
        self.has_label = has_label
        self.train_iid_map_df = train_iid_map_df

    @property
    def _edge_triplets(self):
        triplets_dict = {
            'group_chat': [('src_id', 'group-chat-to', 'src_id'), ('src_id', 'group-chat-by', 'src_id')],
            'ad_click': [('src_id', 'clicked', 'ad_cate'), ('ad_cate', 'clicked-by', 'src_id')],
            'ad_convert': [('src_id', 'converted', 'ad_cate'), ('ad_cate', 'converted-by', 'src_id')],
        }
        if self.has_label:
            triplets_dict.update({
                'label_1': self.pos_label_edge_types,
                'label_0': self.neg_label_edge_types,
        })
        return triplets_dict

    def _verify_has_label(self):
        assert self.has_label, "Your dataset has no labels"

    def _verify_graph_init(self):
        assert self.graph is not None, "Your graph wasn't initialized"

    @property
    def label_edge_types(self):
        return self.pos_label_edge_types + self.neg_label_edge_types

    @property
    def pos_label_edge_types(self):
        self._verify_has_label()
        return [('src_id', 'will-convert', 'ad_cate')]

    @property
    def neg_label_edge_types(self):
        self._verify_has_label()
        return [('src_id', 'will-click', 'ad_cate')]

    @property
    def num_items(self):
        return self.iid_map_df.shape[0]

    @property
    def num_users(self):
        return self.uid_map_df.shape[0]

    @property
    def node2item(self):
        return self.iid_map_df.set_index(f'{self.item_id}_{self.new_id_suffix}')[self.item_id].to_dict()

    @property
    def node2user(self):
        return self.uid_map_df.set_index(f'{self.user_id}_{self.new_id_suffix}')[self.user_id].to_dict()

    @property
    def user2node(self):
        return self.uid_map_df.set_index(self.user_id)[f'{self.user_id}_{self.new_id_suffix}'].to_dict()

    @property
    def item2node(self):
        return self.iid_map_df.set_index(self.item_id)[f'{self.item_id}_{self.new_id_suffix}'].to_dict()

    @property
    def iid_columns(self):
        return self.iid_map_df[self.item_id].tolist()

    @property
    def num_user_features(self):
        return self.feature_dict[self.user_id].shape[1]

    def summary_data(self):
        df_group = self.data_dict['group_chat']
        df_click = self.data_dict['ad_click']
        df_convert = self.data_dict['ad_convert']
        if self.has_label:
            df_label_1 = self.data_dict['label_1']
            df_label_0 = self.data_dict['label_0']
        else:
            # dummy data, to report only
            df_label_1 = pd.DataFrame(columns=[self.user_id, self.item_id])
            df_label_0 = pd.DataFrame(columns=[self.user_id, self.item_id])
        summary = "========================= Data Summary =========================\n" \
                  "- group_chat: #rows = {:8d}\n" \
                  "- ad_click  : #rows = {:8d} | #users = {:8d} | #items = {:3d}\n" \
                  "- ad_convert: #rows = {:8d} | #users = {:8d} | #items = {:3d}\n" \
                  "- label_1   : #rows = {:8d} | #users = {:8d} | #items = {:3d}\n" \
                  "- label_0   : #rows = {:8d} | #users = {:8d} | #items = {:3d}\n" \
                  "- Union     :         {:8s} | #users = {:8d} | #items = {:3d}\n" \
                  "- user_feats:         {:8s} | #users = {:8d} | #feats = {:3d}".format(
            df_group.shape[0],
            df_click.shape[0], df_click[self.user_id].nunique(), df_click[self.item_id].nunique(),
            df_convert.shape[0], df_convert[self.user_id].nunique(), df_convert[self.item_id].nunique(),
            df_label_1.shape[0], df_label_1[self.user_id].nunique(), df_label_1[self.item_id].nunique(),
            df_label_0.shape[0], df_label_0[self.user_id].nunique(), df_label_0[self.item_id].nunique(),
            '', self.uid_map_df.shape[0], self.iid_map_df.shape[0],
            '', self.feature_dict[self.user_id].shape[0], self.feature_dict[self.user_id].shape[1] - 1
        )
        print(summary)

    def init_graph_schema(self):
        graph_schema = dict()
        for data_name, df in self.data_dict.items():
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

    def _import_feature(self, g):
        g.nodes[self.user_id].data['features'] = self.feature_dict[self.user_id]
        g.nodes[self.item_id].data['features'] = self.feature_dict[self.item_id]
        return g

    def init_graph(self):
        graph_schema = self.init_graph_schema()
        num_nodes_dict = {
            self.user_id: self.uid_map_df.shape[0],
            self.item_id: self.iid_map_df.shape[0]
        }
        graph = heterograph(graph_schema, num_nodes_dict=num_nodes_dict)
        graph = self._import_feature(graph)
        self.graph = graph

    def init_adjust_graph(self):
        adjust_graph = self.graph.clone()
        for e_type in self.pos_label_edge_types:
            adjust_graph.remove_edges(torch.arange(self.graph.number_of_edges(e_type)), etype=e_type)
        for e_type in self.neg_label_edge_types:
            adjust_graph.remove_edges(torch.arange(self.graph.number_of_edges(e_type)), etype=e_type)
        self.adjust_graph = adjust_graph

    def _use_train_iid_map(self, df_list, train_iid_map_df):
        for df in df_list:
            df = df.drop(columns=[f'{self.item_id}_{self.new_id_suffix}'])
            df = df.merge(train_iid_map_df, on=self.item_id)
            yield df

    def _verify_all_user_feature_exist(self, user_feature, user_data_list):
        id_set = set()
        _ = [id_set.update(df[id_column]) for df in user_data_list
             for id_column in self.user_ids if id_column in df.columns]
        diff_users = id_set.difference(user_feature[self.user_id])
        assert len(diff_users) == 0, f"There are {len(diff_users)} having no features"

    def load_data(self, print_summary=True):

        user_feature, df_ad, df_group, df_label = self._load_data()

        if df_label is None:
            # dummy data, will not be added to data_dict later
            df_label = pd.DataFrame(columns=[self.user_id, self.item_id, 'label'])

        self._verify_all_user_feature_exist(user_feature=user_feature,
                                            user_data_list=[df_ad, df_group, df_label]
                                            )

        # map src_id and des_id of different dataframes to same indices
        (df_group, df_ad, df_label, user_feature), uid_map_df = create_common_ids(
            [df_group, df_ad, df_label, user_feature], self.user_ids, self.new_id_suffix
        )

        # map ad_cate of different dataframes to same indices
        (df_ad, df_label), iid_map_df = create_common_ids([df_ad, df_label],
                                                          [self.item_id],
                                                          self.new_id_suffix)

        # Ensure index of items in this dataset is the same as that in the training dataset
        if self.train_iid_map_df is not None:
            iid_map_df = self.train_iid_map_df
            df_ad, df_label = self._use_train_iid_map([df_ad, df_label], self.train_iid_map_df)

        data_dict = {
            'group_chat': df_group,
            'ad_click': df_ad[df_ad['kind'] == 100].reset_index(drop=True),
            'ad_convert': df_ad[df_ad['kind'] == 201].reset_index(drop=True),
        }
        if df_label.shape[0] != 0:
            data_dict['label_1'] = df_label[df_label['label'] == 1].reset_index(drop=True)
            data_dict['label_0'] = df_label[df_label['label'] == 0].reset_index(drop=True)

        self.data_dict = data_dict
        self.uid_map_df = uid_map_df
        self.iid_map_df = iid_map_df

        user_feature = user_feature.sort_values(by=f'{self.user_id}_{self.new_id_suffix}', ascending=True) \
            .drop(columns=[self.user_id, f'{self.user_id}_{self.new_id_suffix}'])
        user_feature = torch.tensor(user_feature.values).float()

        item_feature = torch.tensor(list(range(iid_map_df.shape[0]))).long()

        self.feature_dict = {
            self.user_id: user_feature,
            self.item_id: item_feature
        }

        if print_summary:
            self.summary_data()

    def _load_data(self):
        raise NotImplementedError


class TrainDataSet(BaseDataSet):
    def __init__(self, data_dirs, train_iid_map_df=None):
        super(TrainDataSet, self).__init__(has_label=True, train_iid_map_df=train_iid_map_df)
        self.train_iid_map_df = train_iid_map_df
        self.data_dirs = [x.rstrip('/') for x in data_dirs.split(',')]

    def _load_data(self):
        if len(self.data_dirs) > 1:
            df_ad = read_data_from_multiple_dirs(dir_list=self.data_dirs, filename='ad.parquet')
            df_group = read_data_from_multiple_dirs(dir_list=self.data_dirs, filename='group_chat.parquet')
            user_feature = read_data_from_multiple_dirs(dir_list=self.data_dirs, filename='user_features.parquet')
            df_label = read_data_from_multiple_dirs(dir_list=self.data_dirs, filename='label.parquet')
        else:
            # if only 1 data_dir, don't need rename user_id
            data_dir = self.data_dirs[0]
            df_ad = read_data(data_dir + '/ad.parquet')
            df_group = read_data(data_dir + '/group_chat.parquet')
            user_feature = read_data(data_dir + '/user_features.parquet')
            df_label = read_data(data_dir + '/label.parquet')
        return user_feature, df_ad, df_group, df_label


class InferenceDataSet(BaseDataSet):
    def __init__(self, train_iid_map_df, df_group, df_ad, user_feature, to_infer_uid_df=None):
        super(InferenceDataSet, self).__init__(has_label=False)
        self.train_iid_map_df = train_iid_map_df
        self.user_feature = user_feature
        self.df_group = df_group
        self.df_ad = df_ad

        if to_infer_uid_df is not None:
            self.to_infer_uid_df = to_infer_uid_df
            self._verify_all_user_feature_exist(user_feature, [to_infer_uid_df])
        else:
            self.to_infer_uid_df = user_feature[[self.user_id]]

    def _load_data(self):
        return self.user_feature, self.df_ad, self.df_group, None

    @property
    def to_infer_user_node_id(self):
        user2nid = self.user2node
        to_infer_user_id = self.to_infer_uid_df[self.user_id].tolist()
        to_infer_node_id = [user2nid[uid] for uid in to_infer_user_id]
        return to_infer_node_id


class ObjectView(object):
    def __init__(self, d):
        self.__dict__ = d
