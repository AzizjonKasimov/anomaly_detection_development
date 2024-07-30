# -*- coding: utf-8 -*-
"""
Graph Neural Network-Based Anomaly Detection.

References
----------
[1] Deng, Ailin, and Bryan Hooi. "Graph neural network-based anomaly detection in multivariate time series."
Proceedings of the AAAI conference on artificial intelligence. Vol. 35. No. 5. 2021.
[2] Buchhorn, Katie, et al. "Graph Neural Network-Based Anomaly Detection for River Network Systems"
arXiv preprint arXiv:2304.09367 (2023).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax

__author__ = ["KatieBuc"]


class GraphLayer(MessagePassing):
    """
    Class for graph convolutional layers using message passing.

    Attributes
    ----------
    in_channels : int
        Number of input channels for the layer
    out_channels : int
        Number of output channels for the layer
    heads : int
        Number of heads for multi-head attention
    concat_heads : bool
        Whether to concatenate across heads
    negative_slope : float
        Slope for LeakyReLU
    dropout : float
        Dropout rate
    lin : nn.Module
        Linear layer for transforming input
    att_i : nn.Parameter
        Attention parameter related to x_i
    att_j : nn.Parameter
        Attention parameter related to x_j
    att_em_i : nn.Parameter
        Attention parameter related to embedding of x_i
    att_em_j : nn.Parameter
        Attention parameter related to embedding of x_j
    bias : nn.Parameter
        Bias parameter added after message propagation
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        heads=1,
        concat_heads=True,
        negative_slope=0.2,
        dropout=0,
    ):
        super(GraphLayer, self).__init__(aggr="add", node_dim=0)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat_heads = concat_heads
        self.negative_slope = negative_slope
        self.dropout = dropout

        # parameters related to weight matrix W
        self.lin = Linear(in_channels, heads * out_channels, bias=False)

        # attention parameters related to x_i, x_j
        self.att_i = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_j = Parameter(torch.Tensor(1, heads, out_channels))

        # attention parameters related embeddings v_i, v_j
        self.att_em_i = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_em_j = Parameter(torch.Tensor(1, heads, out_channels))

        # if concatenating the across heads, consider the change of out_channels
        self._out_channels = heads * out_channels if concat_heads else out_channels
        self.bias = Parameter(torch.Tensor(self._out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        """Initialise parameters of GraphLayer."""
        glorot(self.lin.weight)
        glorot(self.att_i)
        glorot(self.att_j)
        zeros(self.att_em_i)
        zeros(self.att_em_j)
        self.bias.data.zero_()

    def forward(self, x, edge_index, embedding):
        """Forward method for propagating messages of GraphLayer.

        Parameters
        ----------
        x : tensor
            has shape [N x batch_size, in_channels], where N is the number of nodes
        edge_index : tensor
            has shape [2, E x batch_size], where E is the number of edges
            with E = topk x N
        embedding : tensor
            has shape [N x batch_size, out_channels]
        """
        # linearly transform node feature matrix
        assert torch.is_tensor(x)
        x = self.lin(x)

        # add self loops, nodes are in dim 0 of x
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(self.node_dim))

        # propagate messages
        out = self.propagate(
            edge_index,
            x=(x, x),
            embedding=embedding,
            edges=edge_index,
        )

        # transform [N x batch_size, 1, _out_channels] to [N x batch_size, _out_channels]
        out = out.view(-1, self._out_channels)

        # apply final bias vector
        out += self.bias

        return out

    def message(self, x_i, x_j, edge_index_i, size_i, embedding, edges):
        """Calculate the attention weights using the embedding vector, eq (6)-(8) in [1].

        Parameters
        ----------
        x_i : tensor
            has shape [(topk x N x batch_size), out_channels]
        x_j : tensor
            has shape [(topk x N x batch_size), out_channels]
        edge_index_i : tensor
            has shape [(topk x N x batch_size)]
        size_i : int
            with value (N x batch_size)
        embedding : tensor
            has shape [(N x batch_size), out_channels]
        edges : tensor
            has shape [2, (topk x N x batch_size)]
        """
        # transform to [(topk x N x batch_size), 1, out_channels]
        x_i = x_i.view(-1, self.heads, self.out_channels)
        x_j = x_j.view(-1, self.heads, self.out_channels)

        if embedding is not None:
            # [(topk x N x batch_size), 1, out_channels]
            embedding_i = embedding[edge_index_i].unsqueeze(1).repeat(1, self.heads, 1)
            embedding_j = embedding[edges[0]].unsqueeze(1).repeat(1, self.heads, 1)

            # [(topk x N x batch_size), 1, 2 x out_channels]
            key_i = torch.cat((x_i, embedding_i), dim=-1)
            key_j = torch.cat(
                (x_j, embedding_j), dim=-1
            )  # concatenates along the last dim, i.e. columns in this case

        # concatenate learnable parameters to become [1, 1, 2 x out_channels]
        cat_att_i = torch.cat((self.att_i, self.att_em_i), dim=-1)
        cat_att_j = torch.cat((self.att_j, self.att_em_j), dim=-1)

        # [(topk x N x batch_size), 1]
        alpha = (key_i * cat_att_i).sum(-1) + (key_j * cat_att_j).sum(
            -1
        )  # the matrix multiplication between a^T and g's in eqn (7)

        alpha = alpha.view(-1, self.heads, 1)  # [(topk x N x batch_size), 1, 1]
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, None, size_i)  # eqn (8)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # save to self
        self.alpha = alpha

        # multiply node feature by alpha
        return x_j * alpha

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_channels}, {self.out_channels}, heads={self.heads})"


class OutLayer(nn.Module):
    """
    Output layer used to transform graph layers into a prediction.

    Attributes
    ----------
    mlp : nn.ModuleList
        A module list that contains a sequence of transformations in the output layer
    """

    def __init__(self, in_num, layer_num, inter_dim):
        """
        Parameters
        ----------
        in_num : int
            input dimension of network
        layer_num : int
            number of layers in network
        inter_dim : int
            internal dimensions of layers in network
        """
        super(OutLayer, self).__init__()
        modules = []
        for i in range(layer_num):
            if i == layer_num - 1:
                modules.append(nn.Linear(in_num if layer_num == 1 else inter_dim, 1))
            else:
                layer_in_num = in_num if i == 0 else inter_dim
                modules.extend(
                    (
                        nn.Linear(layer_in_num, inter_dim),
                        nn.BatchNorm1d(inter_dim),
                        nn.ReLU(),
                    )
                )
        self.mlp = nn.ModuleList(modules)

    def forward(self, x):
        """
        Forward pass of output layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        out : torch.Tensor
            Output tensor
        """
        out = x

        for mod in self.mlp:
            if isinstance(mod, nn.BatchNorm1d):
                out = out.permute(0, 2, 1)
                out = mod(out)
                out = out.permute(0, 2, 1)
            else:
                out = mod(out)

        return out


class GNNLayer(nn.Module):
    """
    Calculates the node representations, z_i, in eq (5) of [1].

    Attributes
    ----------
    gnn : GraphLayer
        Graph convolutional layer
    bn : nn.BatchNorm1d
        Batch normalization layer
    relu : nn.ReLU
        ReLU activation function
    """

    def __init__(self, in_channel, out_channel, heads=1):
        """
        Parameters
        ----------
        in_channels : int
            Number of input channels for the layer
        out_channels : int
            Number of output channels for the layer
        heads : int
            Number of heads for multi-head attention
        """
        super(GNNLayer, self).__init__()

        self.gnn = GraphLayer(in_channel, out_channel, heads=heads, concat_heads=False)
        self.bn = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, embedding=None):
        out = self.gnn(x, edge_index, embedding)
        out = self.bn(out)
        return self.relu(out)


class GDN(nn.Module):
    """
    A graph-based network model for time series data, as introduced in [1].

    Attributes
    ----------
    embedding : nn.Embedding
        Node embeddings for the graph
    bn_outlayer_in : nn.BatchNorm1d
        Batch normalization layer applied before the output layer
    gnn_layers : nn.ModuleList
        List of GNNLayer instances used in the network
    learned_graph : tensor
        Topk indices represneting the learned graph, with shape [N, top_k]
    out_layer : OutLayer
        Output layer for the network
    dp : nn.Dropout
        Dropout layer applied before the output layer
    cache_fc_edge_idx : tensor
        has shape [2, (E x batch_size)] where E is the number of edges
    """

    def __init__(
        self,
        fc_edge_idx,
        n_nodes,
        embed_dim=64,
        out_layer_inter_dim=256,
        slide_win=15,
        out_layer_num=1,
        topk=20,
    ):
        """
        Parameters
        ----------
        fc_edge_idx : torch.LongTensor
            Edge indices of fully connected graph for the input time series
        n_nodes : int
            Number of nodes in the graph
        embed_dim : int, optional (default=64)
            Dimension of node embeddings
        out_layer_inter_dim : int, optional (default=256)
            Internal dimensions of layers in the output network
        slide_win : int, optional (default=15)
            Size of sliding window used for input time series
        out_layer_num : int, optional (default=1)
            Number of layers in OutLayer
        topk : int, optional (default=20)
            Number of top-k neighbors to consider when creating learned graph
        """
        super(GDN, self).__init__()
        self.fc_edge_idx = fc_edge_idx
        self.n_nodes = n_nodes
        self.embed_dim = embed_dim
        self.out_layer_inter_dim = out_layer_inter_dim
        self.slide_win = slide_win
        self.out_layer_num = out_layer_num
        self.topk = topk

    def _initialise_layers(self):
        self.embedding = nn.Embedding(self.n_nodes, self.embed_dim)
        self.bn_outlayer_in = nn.BatchNorm1d(self.embed_dim)

        self.gnn_layers = nn.ModuleList(
            [
                GNNLayer(
                    self.slide_win,
                    self.embed_dim,
                    heads=1,
                )
            ]
        )
        self.out_layer = OutLayer(
            self.embed_dim, self.out_layer_num, inter_dim=self.out_layer_inter_dim
        )

        self.dp = nn.Dropout(0.2)
        nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))

    def forward(self, data):
        x = data.clone().detach()
        device = data.device
        batch_size = x.shape[0]

        x = x.view(-1, self.slide_win).contiguous()  # [(batch_size x N), slide_win]

        self.cache_fc_edge_idx = get_batch_edge_index(
            self.fc_edge_idx, batch_size, self.n_nodes
        ).to(device)

        idxs = torch.arange(self.n_nodes, device=device)
        weights = self.embedding(idxs).detach().clone()  # [N, embed_dim]
        batch_embeddings = self.embedding(idxs).repeat(
            batch_size, 1
        )  # [(N x batch_size), embed_dim]

        # e_{ji} in eqn (2)
        cos_ji_mat = torch.matmul(weights, weights.T)  # [N , N]
        normed_mat = torch.matmul(
            weights.norm(dim=-1).view(-1, 1), weights.norm(dim=-1).view(1, -1)
        )
        cos_ji_mat = cos_ji_mat / normed_mat

        # A_{ji} in eqn (3)
        topk_indices_ji = torch.topk(cos_ji_mat, self.topk, dim=-1)[1]
        self.learned_graph = topk_indices_ji  # [N x topk]

        gated_i = (
            torch.arange(0, self.n_nodes)
            .repeat_interleave(self.topk)
            .unsqueeze(0)
            .to(device)
        )  # [N x topk]
        gated_j = topk_indices_ji.flatten().unsqueeze(0)  # [N x topk]
        gated_edge_index = torch.cat((gated_j, gated_i), dim=0)  # [2, (N x topk)]

        batch_gated_edge_index = get_batch_edge_index(
            gated_edge_index, batch_size, self.n_nodes
        ).to(
            device
        )  # [2, (N x topk x batch_size)]

        gcn_out = self.gnn_layers[0](
            x,
            batch_gated_edge_index,
            embedding=batch_embeddings,
        )
        gcn_out = gcn_out.view(
            batch_size, self.n_nodes, -1
        )  # [batch_size, N, embed_dim]

        # eqn (9), element-wise multiply node representation z_i with corresponding embedding v_i
        out = torch.mul(gcn_out, self.embedding(idxs))  # [batch_size, N, embed_dim]
        out = out.permute(0, 2, 1)  # [batch_size, embed_dim, N]
        out = F.relu(self.bn_outlayer_in(out))
        out = out.permute(0, 2, 1)  # [batch_size, N, embed_dim]
        out = self.dp(out)
        out = self.out_layer(out)
        out = out.view(-1, self.n_nodes)  # [batch_size, N]

        return out


def get_batch_edge_index(edge_index, batch_size, n_nodes):
    """
    Replicates neighbour relations for new batch index values.

    Parameters
    ----------
    edge_index : tensor
        has shape [2, E] where E is the number of edges
    batch_size : int
        the size of the batch
    n_nodes : int
        number of nodes, N

    Returns
    -------
    batch_edge_index : tensor
        has shape [2, (E x batch_size)] where E is the number of edges

    Example
    -------
    >>> edge_index = tensor([[0, 2, 1, 2, 2, 1],
                             [0, 0, 1, 1, 2, 2]])
    >>> get_batch_edge_index(edge_index, 2, 3)
    >>> tensor([[0, 2, 1, 2, 2, 1, 3, 5, 4, 5, 5, 4],
                [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]])
    """
    edge_index = edge_index.clone().detach()
    edge_num = edge_index.shape[1]
    batch_edge_index = edge_index.repeat(1, batch_size).contiguous()

    for i in range(batch_size):
        batch_edge_index[:, i * edge_num : (i + 1) * edge_num] += i * n_nodes

    return batch_edge_index.long()