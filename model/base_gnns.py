import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch_geometric.loader import NeighborSampler
from torch_geometric.nn import SAGEConv, GCNConv, GATConv, SGConv
from torch_geometric.nn import inits
from model.base_nns import MLP

from torch_geometric.utils import dropout_adj


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, output_type="logit", adj_dropout=0.0):
        super(GraphSAGE, self).__init__()

        self.num_layers = num_layers
        self.drop_out = dropout
        # self.device = device
        self.output_type = output_type

        self.convs = torch.nn.ModuleList()

        self.normalise = False

        self.ebd_dim = out_channels

        self.adj_dropout = adj_dropout

        if self.num_layers == 1:
            print("Building single-layer GraphSage")
            self.convs.append(SAGEConv(in_channels, out_channels))
        else:
            print("Building multi-layer GraphSage")
            self.convs.append(SAGEConv(in_channels, hidden_channels, normalize=self.normalise))

            for _ in range(num_layers - 2):
                print("One hidden layer added.")
                self.convs.append(SAGEConv(hidden_channels, hidden_channels, normalize=self.normalise))

            self.convs.append(SAGEConv(hidden_channels, out_channels, normalize=self.normalise))

    def reset_parameters(self):
        print("Initialising")
        for conv in self.convs:
            inits.kaiming_uniform(conv.lin_l.weight, conv.lin_l.in_channels, a=math.sqrt(5))
            inits.kaiming_uniform(conv.lin_r.weight, conv.lin_r.in_channels, a=math.sqrt(5))
            inits.zeros(conv.lin_l.in_channels)
            inits.zeros(conv.lin_r.in_channels)
            # conv.reset_parameters()

    def forward(self, x, adjs, pp_matrix=None):
        # `train_loader` computes the k-hop neighborhood of a batch of nodes,
        # and returns, for each layer, a bipartite graph object, holding the
        # bipartite edges `edge_index`, the index `e_id` of the original edges,
        # and the size/shape `size` of the bipartite graph.
        # Target nodes are also included in the source nodes so that one can
        # easily apply skip-connections or add self-loops.
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.

            if self.adj_dropout > 0:
                edge_index = dropout_adj(edge_index, p=self.adj_dropout, force_undirected=True, training=self.training)[0]

            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.drop_out, training=self.training)

        if self.output_type == "ebd":
            # x = F.relu(x)
            # x = F.dropout(x, p=self.drop_out, training=self.training)
            return x

        return x.log_softmax(dim=-1).float()

class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, heads):
        super().__init__()

        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()

        self.skip = None

        # First Layer
        self.convs.append(GATConv(in_channels, hidden_channels, heads))

        for _ in range(num_layers - 2):
            self.convs.append(GATConv(heads * hidden_channels, hidden_channels, heads))

        # Final Layer
        self.convs.append(GATConv(heads * hidden_channels, out_channels, heads, concat=False))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

        if self.skip is not None:
            for skip in self.skips:
                skip.reset_parameters()

    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)

        return x


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=True)
        self.conv2 = GCNConv(hidden_channels, out_channels, cached=True)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return

class train_model(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        input_dim = args.input_dim
        hidden_dim = args.hidden_dim
        n_layers = args.n_layers
        drop_out = args.drop_out
        ebd_dim = args.ebd_dim
        self.encoder = GraphSAGE(input_dim, hidden_dim, hidden_dim, n_layers, drop_out, output_type="ebds", adj_dropout=0.0)
        self.linear = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.ReLU(),
                nn.Dropout(drop_out)
            ),
            nn.Linear(input_dim, hidden_dim)
        ])
        self.lin1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.proj = MLP(ebd_dim, ebd_dim, ebd_dim)
        if args.loss == 'bce':
            self.clf = MLP(ebd_dim, 32, 1)
        elif args.loss == 'dev':
            self.clf = torch.nn.Linear(hidden_dim, 1)
    def forward(self, x, adjs, ppr_matrix=None, loss_type='consistency', return_ebds=False):
        ebds = self.encoder(x, adjs, ppr_matrix)
        if ppr_matrix is not None:
            # div_loss
            out_1 = F.normalize(self.lin1(ebds), dim=-1)
            out_2 = F.normalize(self.lin2(ebds), dim=-1)
            div_loss = ((out_1 - out_2) ** 2).sum(dim=-1).mean()
            # return ebds, div_loss
            for i, (edge_index, _, size) in enumerate(adjs):
                x_target = x[:size[1]]
                ppr_matrix_target = ppr_matrix[:size[1], :size[1]]
                x = ppr_matrix_target @ x_target # H， similarity matrix
                x = self.linear[i](x)
            graph1 = F.normalize(ebds, dim=-1) # for x and adj
            graph2 = F.normalize(x, dim=-1) # for x and ppr_matrix
            if loss_type == 'contrastive':
                logits = graph1 @ graph2.t()
                return logits, div_loss
            elif loss_type == 'consistency':
                embed1 = self.proj(graph1)
                embed2 = self.proj(graph2)
                logits1 = self.clf(embed1)
                logits2 = self.clf(embed2)
                return (logits1, logits2), div_loss
        ebds = self.proj(ebds)
        logits = self.clf(ebds)
        if return_ebds:
            return logits, ebds
        else:
            return logits

    def energy(self, logits):
        return -torch.logsumexp(logits, dim=1)