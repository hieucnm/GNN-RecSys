from typing import Tuple

import dgl.function as dglfn
import dgl.nn.pytorch as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F


def udf_u_cat_e(edges):
    # TODO: what if we add a normalize step after concat ??
    return {'m': torch.cat([edges.src['h'], edges.data['h']], 1)}


class Normalize(nn.Module):
    def forward(self, x):
        x_norm = x.norm(2, 1, keepdim=True)
        x_norm = torch.where(x_norm == 0, torch.tensor(1.).to(x_norm), x_norm)
        return x / x_norm


class LinearWithBatchNorm(nn.Module):
    def __init__(self, in_feats, out_feats, bias=False):
        super(LinearWithBatchNorm, self).__init__()
        self.bn = nn.BatchNorm1d(in_feats)
        self.ln = nn.Linear(in_feats, out_feats, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        torch.random.seed()
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.ln.weight, gain=gain)

    def forward(self, x):
        return self.ln(self.bn(x))


class LinearWithNormalize(nn.Module):
    def __init__(self, in_feats, out_feats, bias=False):
        super(LinearWithNormalize, self).__init__()
        self.ln = nn.Linear(in_feats, out_feats, bias=bias)
        self.norm = Normalize()
        self.relu = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        torch.random.seed()
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.ln.weight, gain=gain)

    def forward(self, x):
        return self.norm(self.relu(self.ln(x)))


class ConvLayer(nn.Module):
    """
    1 layer of message passing & aggregation, specific to an edge type.
    Similar to SAGEConv layer in DGL library, but advanced.
    """
    def __init__(self,
                 in_feats: Tuple[int, int],
                 out_feats: int,
                 aggregator_type: str,
                 dropout: float,
                 norm: bool = True,
                 pre_aggregate: bool = False,
                 use_edge_feat: bool = False,
                 edge_agg_type: str = None,
                 edge_feats: int = None,
                 ):
        super().__init__()

        assert aggregator_type in ['max', 'mean', 'sum'], 'Aggregator type {} not recognized.'.format(aggregator_type)
        if use_edge_feat:
            assert edge_agg_type in ['u_mul_e', 'u_add_e', 'u_cat_e'], 'Edge aggregator type {} not recognized.'.format(
                edge_agg_type)

        self.relu = nn.ReLU()
        self.dropout_fn = nn.Dropout(dropout)
        self.normalize = Normalize() if norm else nn.Identity()
        self.pre_aggregate = pre_aggregate
        self.use_edge_feat = use_edge_feat
        self.edge_agg_type = edge_agg_type

        self.agg_func = getattr(dglfn, aggregator_type)  # E.g: getattr(fn, 'mean') equals to fn.mean
        self.msgs_func = self._get_msgs_func()

        in_neigh_feats, in_self_feats = in_feats
        self.fc_self = nn.Linear(in_self_feats, out_feats, bias=False)
        self.fc_neigh = nn.Linear(in_neigh_feats, out_feats, bias=False)

        # pre-aggregate
        if pre_aggregate:
            self.fc_preagg = nn.Linear(in_neigh_feats, in_neigh_feats, bias=False)

        # edges
        if use_edge_feat:
            # self.fc_edge = LinearWithBatchNorm(edge_feats, in_neigh_feats, bias=False)
            self.fc_edge = LinearWithNormalize(edge_feats, in_neigh_feats, bias=False)
            if edge_agg_type == 'u_cat_e':
                self.fc_remap = nn.Linear(in_neigh_feats * 2, in_neigh_feats, bias=False)

        self.reset_parameters()

    def _get_msgs_func(self):
        if self.use_edge_feat:
            if self.edge_agg_type == 'u_cat_e':
                return udf_u_cat_e
            return getattr(dglfn, self.edge_agg_type)('h', 'h', 'm')
        return dglfn.copy_src('h', 'm')

    def reset_parameters(self):
        torch.random.seed()
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)
        if self.use_edge_feat:
            nn.init.xavier_uniform_(self.fc_edge.ln.weight, gain=gain)
            if self.edge_agg_type == 'u_cat_e':
                nn.init.xavier_uniform_(self.fc_remap.weight, gain=gain)
        if self.pre_aggregate:
            nn.init.xavier_uniform_(self.fc_preagg.weight, gain=gain)

    def forward(self,
                graph,
                x):
        h_neigh, h_self = x
        h_neigh = self.dropout_fn(h_neigh)
        h_self = self.dropout_fn(h_self)

        # message passing
        graph.srcdata['h'] = self.relu(self.fc_preagg(h_neigh)) if self.pre_aggregate else h_neigh
        if self.use_edge_feat:
            graph.edata['h'] = self.fc_edge(graph.edata['features'])
        graph.update_all(self.msgs_func, self.agg_func('m', 'neigh'))
        h_neigh = graph.dstdata['neigh']

        # if yes, dimension of h_neigh was doubled due to concatenation
        if self.use_edge_feat and self.edge_agg_type == 'u_cat_e':
            h_neigh = self.relu(self.fc_remap(h_neigh))

        z = self.relu(self.fc_self(h_self) + self.fc_neigh(h_neigh))
        z = self.normalize(z)
        return z


class PredictingLayer(nn.Module):
    """
    Scoring function that uses a neural network to compute similarity between user and item.

    Only used if fixed_params.pred == 'sigmoid'.
    Given the concatenated hidden states of heads and tails vectors, passes them through neural network and
    returns sigmoid ratings.
    """

    def reset_parameters(self):
        gain_relu = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.hidden.weight, gain=gain_relu)
        gain_sigmoid = nn.init.calculate_gain('sigmoid')
        nn.init.xavier_uniform_(self.output.weight, gain=gain_sigmoid)

    def __init__(self, embed_dim: int):
        super(PredictingLayer, self).__init__()
        self.hidden = nn.Linear(embed_dim * 2, 32)
        self.output = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.reset_parameters()

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x


class PredictingModule(nn.Module):
    """
    Predicting module that incorporate the predicting layer defined earlier.

    Only used if fixed_params.pred == 'nn'.
    Process:
        - Fetches hidden states of 'heads' and 'tails' of the edges.
        - Concatenates them, then passes them through the predicting layer.
        - Returns ratings (sigmoid function).
    """

    def __init__(self,
                 predicting_layer,
                 embed_dim: int,
                 train_nodes,
                 ):
        super(PredictingModule, self).__init__()
        self.layer_nn = predicting_layer(embed_dim)
        self.train_nodes = train_nodes

    def get_ratings(self, x, y):
        cat_embed = torch.cat((x, y), 1)
        return self.layer_nn(cat_embed)

    def forward(self,
                graph,
                h
                ):
        ratings_dict = {}
        for edge_type in graph.canonical_etypes:
            src_node, _, dst_node = edge_type
            if src_node in self.train_nodes and dst_node in self.train_nodes:
                src_nid, dst_nid = graph.all_edges(etype=edge_type)
                src_emb = h[src_node][src_nid]
                dst_emb = h[dst_node][dst_nid]
                cat_embed = torch.cat((src_emb, dst_emb), 1)
                ratings = self.layer_nn(cat_embed)
                ratings_dict[edge_type] = torch.flatten(ratings)
        ratings_dict = {key: torch.unsqueeze(ratings_dict[key], 1) for key in ratings_dict.keys()}
        return ratings_dict


class CosinePrediction(nn.Module):
    """
    Scoring function that uses cosine similarity to compute similarity between user and item.

    Only used if fixed_params.pred == 'cos'.
    """

    def __init__(self):
        super().__init__()

    def get_ratings(self, x, y):
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        return cos(x, y)

    def forward(self, graph, h):
        with graph.local_scope():
            for etype in graph.canonical_etypes:
                try:
                    graph.nodes[etype[0]].data['norm_h'] = F.normalize(h[etype[0]], p=2, dim=-1)
                    graph.nodes[etype[2]].data['norm_h'] = F.normalize(h[etype[2]], p=2, dim=-1)
                    graph.apply_edges(dglfn.u_dot_v('norm_h', 'norm_h', 'cos'), etype=etype)
                except KeyError:
                    pass  # For etypes that are not in training eids, thus have no 'h'
            ratings = graph.edata['cos']
        return ratings


class ConvModel(nn.Module):
    """
    Assembles embedding layers, multiple ConvLayers and chosen predicting function into a full model.

    """

    def __init__(self,
                 edge_types,
                 dim_dict,
                 user_id: str,
                 item_id: str,
                 pred: str,
                 n_layers: int,
                 aggregator_hetero: str,
                 aggregator_homo: str,
                 norm: bool = True,
                 dropout: float = 0.0,
                 pre_aggregate: bool = False,
                 use_edge_feat: bool = False,
                 edge_agg_type: str = None,
                 edge_feats_dict: dict = None,
                 ):
        """
        Initialize the ConvModel.

        Parameters
        ----------
        edge_types:
            List of edge types to create hetero layers
        user_id, item_id:
            Name of user node and item node in the graph to get the features correctly
        n_layers:
            Number of ConvLayer.
        dim_dict:
            Dictionary with dimension for all input nodes, hidden dimension (aka embedding dimension), output dimension.
        pred:
            Type of prediction layer. Choices: ['cos', 'pred']
        aggregator_hetero:
            Since we are working with heterogeneous graph, all nodes will have messages coming from different types of
            nodes. However, the neighborhood messages are specific to a node type. Thus, we have to aggregate
            neighborhood messages from different edge types.
            Choices are 'mean', 'sum', 'max'.
        Other Parameters:
            See ConvLayer for details.
        """
        super().__init__()
        self.embed_dim = dim_dict['out']
        self.user_id = user_id
        self.item_id = item_id

        # input layer
        self.user_embed = LinearWithBatchNorm(dim_dict['user'], dim_dict['hidden'])
        self.item_embed = nn.Embedding(dim_dict['item'], dim_dict['hidden'])

        # hidden layers
        common_conv_params = {
            'in_feats': (dim_dict['hidden'], dim_dict['hidden']),
            'aggregator_type': aggregator_homo,
            'dropout': dropout,
            'norm': norm,
            'pre_aggregate': pre_aggregate,
            'use_edge_feat': use_edge_feat,
            'edge_agg_type': edge_agg_type,
        }

        if use_edge_feat:
            assert edge_feats_dict, "`use_edge_feat` is True but `edge_feats_dict` not given"

        self.layers = nn.ModuleList()
        for i in range(n_layers - 1):
            out_feats = dim_dict['out'] if i == n_layers - 2 else dim_dict['hidden']
            self.layers.append(dglnn.HeteroGraphConv({
                e_type[1]: ConvLayer(
                    edge_feats=edge_feats_dict[e_type] if use_edge_feat else None,
                    out_feats=out_feats,
                    **common_conv_params)
                for e_type in edge_types},
                aggregate=aggregator_hetero
            ))

        if pred == 'cos':
            self.pred_fn = CosinePrediction()
        elif pred == 'sigmoid':
            self.pred_fn = PredictingModule(PredictingLayer, dim_dict['out'], (user_id, item_id))
        else:
            raise KeyError('Prediction function {} not recognized.'.format(pred))

    def get_repr(self,
                 blocks,
                 h):
        h[self.user_id] = self.user_embed(h[self.user_id])
        h[self.item_id] = self.item_embed(h[self.item_id])
        for i in range(len(blocks)):
            layer = self.layers[i]
            h = layer(blocks[i], h)
        return h

    def get_ratings(self, x, y):
        return self.pred_fn.get_ratings(x, y)

    def forward(self,
                blocks,
                h,
                pos_g,
                neg_g
                ):
        """
        Full pass through the ConvModel.

        Process:
            - get_repr: As many ConvLayer as wished. All "Layers" are composed of as many ConvLayer as there
                        are edge types.
            - Predicting layer predicts score for all positive examples and all negative examples.

        Parameters
        ----------
        blocks:
            Computational blocks. Can be thought of as subgraphs. There are as many blocks as there are layers.
        h:
            Initial state of all nodes.
        pos_g:
            Positive graph, generated by the EdgeDataLoader. Contains all positive examples of the batch that need to
            be scored.
        neg_g:
            Negative graph, generated by the EdgeDataLoader. For all positive pairs in the pos_g, multiple negative
            pairs were generated. They are all scored.

        Returns
        -------
        h:
            Updated state of all nodes
        pos_score:
            All scores between positive examples (aka positive pairs).
        neg_score:
            All score between negative examples.

        """

        h = self.get_repr(blocks, h)
        pos_score = self.pred_fn(pos_g, h)
        neg_score = self.pred_fn(neg_g, h)
        return h, pos_score, neg_score
