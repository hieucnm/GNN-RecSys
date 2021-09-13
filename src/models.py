from typing import Tuple

import dgl.function as fn
import dgl.nn.pytorch as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F


class NodeEmbedding(nn.Module):
    """
    Projects the node features into embedding space.
    If use_id = True, the ids of nodes will be use to get embeddings.
    Otherwise, the node features will be projected to the embedding space,
        and a batch_norm layer also be used to normalize the features
    """

    def __init__(self,
                 in_feats,
                 out_feats,
                 use_id
                 ):
        super().__init__()
        self.use_id = use_id
        if use_id:
            self.embedding = nn.Embedding(in_feats, out_feats)
        else:
            self.bn = nn.BatchNorm1d(in_feats)
            self.embedding = nn.Linear(in_feats, out_feats)

    def forward(self, node_feats):
        if not self.use_id:
            node_feats = self.bn(node_feats)
        x = self.embedding(node_feats)
        return x


class ConvLayer(nn.Module):
    """
    1 layer of message passing & aggregation, specific to an edge type.

    Similar to SAGEConv layer in DGL library.

    Methods
    -------
    reset_parameters:
        Intialize parameters for all neural networks in the layer.
    _lstm_reducer:
        Aggregate messages of neighborhood nodes using LSTM technique. (All other message aggregation are builtin
        functions of DGL).
    forward:
        Actual message passing & aggregation, & update of nodes messages.

    """

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)
        # nn.init.xavier_uniform_(self.fc_edge.weight, gain=gain)
        if self._aggre_type in ['max_nn', 'max_nn_edge', 'mean_nn', 'mean_nn_edge']:
            nn.init.xavier_uniform_(self.fc_preagg.weight, gain=gain)
        if self._aggre_type == 'lstm':
            self.lstm.reset_parameters()

    def __init__(self,
                 in_feats: Tuple[int, int],
                 out_feats: int,
                 dropout: float,
                 aggregator_type: str,
                 norm,
                 ):
        """
        Initialize the layer with parameters.

        Parameters
        ----------
        in_feats:
            Dimension of the message (aka features) of the node type neighbor and of the node type. E.g. if the
            ConvLayer is initialized for the edge type (user, clicks, item), in_feats should be
            (dimension_of_item_features, dimension_of_user_features). Note that usually features will first go
            through embedding layer, thus dimension might be equal to the embedding dimension.
        out_feats:
            Dimension that the output (aka the updated node message) should take. E.g. if the layer is a hidden layer,
            out_feats should be hidden_dimension, whereas if the layer is the output layer, out_feats should be
            output_dimension.
        dropout:
            Traditional dropout applied to input features.
        aggregator_type:
            This is the main parameter of ConvLayer; it defines how messages are passed and aggregated. Multiple
            choices:
                'mean' : messages are passed normally, and aggregated by doing the mean of all neighbor messages.
                'mean_nn' : messages are passed through a neural network before being passed to neighbors, and
                            aggregated by doing the mean of all neighbor messages.
                'max_nn' : messages are passed through a neural network before being passed to neighbors, and
                            aggregated by doing the max of all neighbor messages.
                'lstm' : messages are passed normally, and aggregared using _lstm_reducer.
            All choices have also their equivalent that ends with _edge (e.g. mean_nn_edge). Those version include
            the edge in the message passing, i.e. the message will be multiplied by the value of the edge.
        norm:
            Apply normalization
        """
        super().__init__()
        self._in_neigh_feats, self._in_self_feats = in_feats
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
        self.dropout_fn = nn.Dropout(dropout)
        self.norm = norm
        self.fc_self = nn.Linear(self._in_self_feats, out_feats, bias=False)
        self.fc_neigh = nn.Linear(self._in_neigh_feats, out_feats, bias=False)
        # self.fc_edge = nn.Linear(1, 1, bias=True)  # Projecting recency days
        if aggregator_type in ['max_nn', 'max_nn_edge', 'mean_nn', 'mean_nn_edge']:
            self.fc_preagg = nn.Linear(self._in_neigh_feats, self._in_neigh_feats, bias=False)
        if aggregator_type == 'lstm':
            self.lstm = nn.LSTM(self._in_neigh_feats, self._in_neigh_feats, batch_first=True)
        self.reset_parameters()

    def forward(self,
                graph,
                x):
        """
        Message passing & aggregation, & update of node messages.

        Process is the following:
            - Messages (h_neigh and h_self) are extracted from x
            - Dropout is applied
            - Messages are passed and aggregated using the _aggre_type specified (see __init__ for details), which
              return updated h_neigh
            - h_self passes through neural network & updated h_neigh passes through neural network, and are then summed
              up
            - The sum (z) passes through Relu
            - Normalization is applied
        """
        h_neigh, h_self = x
        h_neigh = self.dropout_fn(h_neigh)
        h_self = self.dropout_fn(h_self)

        if self._aggre_type == 'mean':
            graph.srcdata['h'] = h_neigh
            graph.update_all(
                fn.copy_src('h', 'm'),
                fn.mean('m', 'neigh'))
            h_neigh = graph.dstdata['neigh']

        elif self._aggre_type == 'max':
            graph.srcdata['h'] = h_neigh
            graph.update_all(
                fn.copy_src('h', 'm'),
                fn.max('m', 'neigh'))
            h_neigh = graph.dstdata['neigh']

        elif self._aggre_type == 'mean_nn':
            graph.srcdata['h'] = F.relu(self.fc_preagg(h_neigh))
            graph.update_all(
                fn.copy_src('h', 'm'),
                fn.mean('m', 'neigh'))
            h_neigh = graph.dstdata['neigh']

        elif self._aggre_type == 'max_nn':
            graph.srcdata['h'] = F.relu(self.fc_preagg(h_neigh))
            graph.update_all(
                fn.copy_src('h', 'm'),
                fn.max('m', 'neigh'))
            h_neigh = graph.dstdata['neigh']

        elif self._aggre_type == 'mean_edge':
            graph.srcdata['h'] = h_neigh
            if graph.canonical_etypes[0][0] in ['user', 'item'] and graph.canonical_etypes[0][2] in ['user', 'item']:
                graph.edata['h'] = graph.edata['occurrence'].float().unsqueeze(1)
                graph.update_all(
                    fn.u_mul_e('h', 'h', 'm'),
                    fn.mean('m', 'neigh'))
            else:
                graph.update_all(
                    fn.copy_src('h', 'm'),
                    fn.mean('m', 'neigh'))
            h_neigh = graph.dstdata['neigh']

        elif self._aggre_type == 'mean_nn_edge':
            graph.srcdata['h'] = F.relu(self.fc_preagg(h_neigh))
            if graph.canonical_etypes[0][0] in ['user', 'item'] and graph.canonical_etypes[0][2] in ['user', 'item']:
                graph.edata['h'] = graph.edata['occurrence'].float().unsqueeze(1)
                graph.update_all(
                    fn.u_mul_e('h', 'h', 'm'),
                    fn.mean('m', 'neigh'))
            else:
                graph.update_all(
                    fn.copy_src('h', 'm'),
                    fn.mean('m', 'neigh'))
            h_neigh = graph.dstdata['neigh']

        elif self._aggre_type == 'max_nn_edge':
            graph.srcdata['h'] = F.relu(self.fc_preagg(h_neigh))
            if graph.canonical_etypes[0][0] in ['user', 'item'] and graph.canonical_etypes[0][2] in ['user', 'item']:
                graph.edata['h'] = graph.edata['occurrence'].float().unsqueeze(1)
                graph.update_all(
                    fn.u_mul_e('h', 'h', 'm'),
                    fn.max('m', 'neigh'))
            else:
                graph.update_all(
                    fn.copy_src('h', 'm'),
                    fn.max('m', 'neigh'))
            h_neigh = graph.dstdata['neigh']

        else:
            raise KeyError('Aggregator type {} not recognized.'.format(self._aggre_type))

        z = self.fc_self(h_self) + self.fc_neigh(h_neigh)
        z = F.relu(z)

        # normalization
        if self.norm:
            z_norm = z.norm(2, 1, keepdim=True)
            z_norm = torch.where(z_norm == 0,
                                 torch.tensor(1.).to(z_norm),
                                 z_norm)
            z = z / z_norm

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
                    graph.apply_edges(fn.u_dot_v('norm_h', 'norm_h', 'cos'), etype=etype)
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
                 aggregator_homo: str,
                 aggregator_hetero: str,
                 n_layers: int,
                 norm: bool = True,
                 dropout: float = 0.0,
                 ):
        """
        Initialize the ConvModel.

        Parameters
        ----------
        graph:
            Graph, only used to query graph metastructure (fetch node types and edge types).
        n_layers:
            Number of ConvLayer.
        dim_dict:
            Dictionary with dimension for all input nodes, hidden dimension (aka embedding dimension), output dimension.
        norm, dropout, aggregator_homo:
            See ConvLayer for details.
        aggregator_hetero:
            Since we are working with heterogeneous graph, all nodes will have messages coming from different types of
            nodes. However, the neighborhood messages are specific to a node type. Thus, we have to aggregate
            neighborhood messages from different edge types.
            Choices are 'mean', 'sum', 'max'.
        """
        super().__init__()
        self.embed_dim = dim_dict['out']
        self.user_id = user_id
        self.item_id = item_id

        # input layer
        self.user_embed = NodeEmbedding(dim_dict['user'], dim_dict['hidden'], use_id=False)
        self.item_embed = NodeEmbedding(dim_dict['item'], dim_dict['hidden'], use_id=True)

        self.layers = nn.ModuleList()
        # hidden layers
        for i in range(n_layers - 2):
            self.layers.append(dglnn.HeteroGraphConv({
                e_type[1]:
                    ConvLayer(
                         in_feats=(dim_dict['hidden'], dim_dict['hidden']),
                         out_feats=dim_dict['hidden'],
                         dropout=dropout,
                         aggregator_type=aggregator_homo,
                         norm=norm)
                for e_type in edge_types},
                aggregate=aggregator_hetero
            ))

        # output layer
        self.layers.append(dglnn.HeteroGraphConv({
            e_type[1]:
                ConvLayer(
                    in_feats=(dim_dict['hidden'], dim_dict['hidden']),
                    out_feats=dim_dict['out'],
                    dropout=dropout,
                    aggregator_type=aggregator_homo,
                    norm=norm)
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
            - Embedding layer
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
