import torch
from dgl import heterograph

from custom.utils_data import create_common_ids, read_data


class DataSet:
    def __init__(self, data_dir):
        self.new_id_suffix = 'idx'  # e.g: `src_id` will be map to `src_id_idx`
        self.homo_data_names = ['group_chat']  # these data contains user - user interactions

        data_dir = data_dir.rstrip('/')
        self.ad_path = data_dir + '/ad.parquet'
        self.label_path = data_dir + '/label.parquet'
        self.group_chat_path = data_dir + '/group_chat.parquet'
        self.user_feature_path = data_dir + '/user_features.parquet'

        self.data_dict, self.uid_map_df, self.aid_map_df = self.init_data()
        self.feature_dict = self.load_features()
        self.label_edge_types = self.edge_triplets['label']

    @property
    def user_id(self):
        return 'src_id'

    @property
    def item_id(self):
        return 'ad_cate'

    @property
    def edge_triplets(self):
        return {
            'group_chat': [('src_id', 'group-chat-to', 'src_id'), ('src_id', 'group-chat-by', 'src_id')],
            'ad_click': [('src_id', 'clicked', 'ad_cate'), ('ad_cate', 'clicked-by', 'src_id')],
            'ad_convert': [('src_id', 'converted', 'ad_cate'), ('ad_cate', 'converted-by', 'src_id')],
            'label': [('src_id', 'will-convert', 'ad_cate')],
        }

    @property
    def num_items(self):
        return self.aid_map_df.shape[0]

    @property
    def num_user_features(self):
        return self.uid_map_df.shape[1]

    def init_data(self):

        df_group = read_data(self.group_chat_path)  # group chat
        df_ad = read_data(self.ad_path)             # oa_form: both click and convert
        df_label = read_data(self.label_path)       # oa_form in future, convert only

        # map src_id and des_id of different dataframes to same indices
        (df_group, df_ad, df_label), uid_map_df = create_common_ids([df_group, df_ad, df_label],
                                                                    ['src_id', 'des_id'],
                                                                    self.new_id_suffix)
        # map ad_cate of different dataframes to same indices
        (df_ad, df_label), aid_map_df = create_common_ids([df_ad, df_label],
                                                          ['ad_cate'],
                                                          self.new_id_suffix)
        data_dict = {
            'group_chat': df_group,
            'ad_click': df_ad[df_ad['kind'] == 100].reset_index(drop=True),
            'ad_convert': df_ad[df_ad['kind'] == 201].reset_index(drop=True),
            'label': df_label
        }
        return data_dict, uid_map_df, aid_map_df

    def load_features(self):

        feature_dict = {}

        # user feature
        user_feature = read_data(self.user_feature_path)
        user_feature = user_feature.merge(self.uid_map_df, on=self.user_id)
        assert user_feature.shape[0] == self.uid_map_df.shape[0], \
            "no. users in user_feat ({}) not equal no. users in interactions ({})".format(user_feature.shape[0],
                                                                                          self.uid_map_df.shape[0])
        feature_dict[self.user_id] = torch.tensor(user_feature.values).float()

        # item feature: index only
        item_index = self.aid_map_df[f'{self.item_id}_{self.new_id_suffix}']
        feature_dict[self.item_id] = torch.tensor(item_index.values).long()

        # also load other node's features here if necessary
        return feature_dict

    def init_graph_schema(self):

        graph_schema = dict()
        for data_name, df in self.data_dict.items():
            if data_name in self.homo_data_names:
                edge_type = self.edge_triplets[data_name][0]
                reverse_edge_type = self.edge_triplets[data_name][1]
                src_node = f'src_id_{self.new_id_suffix}'
                dst_node = f'des_id_{self.new_id_suffix}'
                graph_schema[edge_type] = (df[src_node].values, df[dst_node].values)
                graph_schema[reverse_edge_type] = (df[dst_node].values, df[src_node].values)
            else:
                for edge_type in self.edge_triplets[data_name]:
                    src_node, _, dst_node = edge_type
                    src_node = f'{src_node}_{self.new_id_suffix}'
                    dst_node = f'{dst_node}_{self.new_id_suffix}'
                    graph_schema[edge_type] = (df[src_node].values, df[dst_node].values)
        return graph_schema

    def init_graph(self):
        graph_schema = self.init_graph_schema()
        graph = heterograph(graph_schema)

        # import features
        graph.nodes[self.user_id].data['features'] = self.feature_dict[self.user_id]
        graph.nodes[self.item_id].data['features'] = self.feature_dict[self.item_id]
        return graph


class ObjectView(object):
    def __init__(self, d):
        self.__dict__ = d
