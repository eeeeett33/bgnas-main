
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):

        super(MLP, self).__init__()

        self.linear_or_not = True
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            return self.linear(x)
        else:
            h = x
            for layer in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
            return self.linears[self.num_layers - 1](h)

class GIN(nn.Module):
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim, output_dim, final_dropout, learn_eps, graph_pooling_type, neighbor_pooling_type):

        super(GIN, self).__init__()

        self.final_dropout = final_dropout
        self.num_layers = num_layers
        self.graph_pooling_type = graph_pooling_type
        self.neighbor_pooling_type = neighbor_pooling_type
        self.learn_eps = learn_eps
        self.eps = nn.Parameter(torch.zeros(self.num_layers-1))

        self.mlps = torch.nn.ModuleList()

        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers):
            if layer == 0:
                self.mlps.append(MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim))
            else:
                self.mlps.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))

            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        self.linears_prediction = torch.nn.ModuleList()
        self.linear_prediction = MLP(num_mlp_layers, input_dim+(num_layers)*hidden_dim, hidden_dim, output_dim)

    def next_layer(self, h, layer, g = None, value=None):

        pooled = dgl.ops.gspmm(g, 'mul', 'sum', lhs_data=h, rhs_data=value)

        pooled_rep = self.mlps[layer](pooled)
        h = self.batch_norms[layer](pooled_rep)
        h = F.relu(h)
        return h

    def forward(self, feat, adj_tensor):
        hidden_rep = [feat]
        h = feat
        try:
            row, column = adj_tensor.coalesce().indices()
            g = dgl.graph((column, row), num_nodes=adj_tensor.shape[0], device=adj_tensor.device)
            value = adj_tensor.coalesce().values()
        except:
            row, column, value = adj_tensor.coo()
            g = dgl.graph((column, row), num_nodes=adj_tensor.size(0), device=adj_tensor.device())

        for layer in range(self.num_layers):
            if layer == 0:
                h = self.mlps[layer](h)
            else:
                h = self.next_layer(h, layer,g = g,value=value)
            hidden_rep.append(h)

        hidden_rep = F.dropout(torch.cat(hidden_rep, 1), self.final_dropout, training=self.training)
        output = self.linear_prediction(hidden_rep)

        return hidden_rep, output
