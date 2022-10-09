import torch

import torch.nn as nn
from torch_scatter import scatter_add, scatter_mean, scatter_max

from ..utils import euclidean_feats, make_mlp
from ..gnn_base import GNNBase

class EB(nn.Module):
    # TODO: Add support for scalar quantities
    def __init__(self, n_edge_attributes: int = 2, n_scalar_attributes: int = 32, n_free_attributes: int = 2, n_equi_hidden: int = 32, n_free_hidden: int = 2, nb_node_layer : int = 2, c_weight: float = 1.0) -> None:
        super(EB, self).__init__()

        # Controls the scale of x during updates
        self.c_weight = c_weight

        self.phi_e = make_mlp(
            n_edge_attributes,
            [n_equi_hidden] * nb_node_layer,
            layer_norm=True,
        )

        # MLP to generate attention weights
        self.phi_x = make_mlp(
            n_equi_hidden,
            [n_equi_hidden] * nb_node_layer + [1],
            layer_norm=True,
        )

        # MLP to generate weights for the messages
        self.phi_m = make_mlp(
            n_equi_hidden,
            [n_equi_hidden] * nb_node_layer,
            output_activation="Sigmoid",
            layer_norm=True,
        )

        self.phi_s = make_mlp(
            n_scalar_attributes,
            [n_equi_hidden] * nb_node_layer,
            layer_norm=True,
        )

        self.phi_h_node = make_mlp(
            n_free_attributes,
            [n_free_hidden] * nb_node_layer,
            layer_norm=True,
        )

        self.phi_h_edge = make_mlp(
            2*n_free_hidden,
            [n_free_hidden] * nb_node_layer,
            layer_norm=True,
        )



    def message(self, norms, dots, s_cat=None, e=None):
        if s_cat is not None and e is not None:
            e_ij = torch.cat([norms, dots, s_cat, e], dim=1)
        elif s_cat is not None:
            e_ij = torch.cat([norms, dots, s_cat], dim=1)
        else:
            e_ij = torch.cat([norms, dots], dim=1)
        e_ij = self.phi_e(e_ij) # The edge features
        m_ij = self.phi_m(e_ij)
        return m_ij, e_ij

    def x_model(self, x, edge_index, x_diff, m):
        i, j = edge_index
        update_val = x_diff * self.phi_x(m)
        # LorentzNet authors clamp the update tensor as a precautionary measure
        update_val = torch.clamp(update_val, min=-100, max=100)
        x_agg = scatter_add(update_val, i, dim=0, dim_size=x.size(0))
        x = x + x_agg * self.c_weight
        return x

    def s_model(self, s, v, edge_index, m):
        i, j = edge_index
        s_agg = scatter_add(m, i, dim=0, dim_size=v.size(0))
        if s is not None:
            s_agg = self.phi_s(torch.cat([s, s_agg], dim=1))
        else:
            s_agg = self.phi_s(s_agg)
        return s_agg

    def forward(self, v, edge_index, s=None, h=None, e=None):
        norms, dots, v_diff, s_cat = euclidean_feats(edge_index, v, s)
        m, e = self.message(norms, dots, s_cat, e)
        v_tilde = self.x_model(v, edge_index, v_diff, m)
        s_tilde = self.s_model(s, v, edge_index, m)

        h_tilde = self.h_model(s_tilde, v_tilde, e, h, edge_index)

        return v_tilde, s_tilde, h_tilde, e

    def h_model(self, s, v, e, h, edge_index):
        i, j = edge_index
        if h is not None:
            h_node_feats = torch.cat([s, v, h], dim=1)
        else:
            h_node_feats = torch.cat([s, v], dim=1)
        h_edge_feats = torch.cat([h_node_feats[i], h_node_feats[j], e], dim=1)
        h_edge_feats = self.phi_h_edge(h_edge_feats)
        h_agg = scatter_add(h_edge_feats, i, dim=0, dim_size=h_node_feats.size(0))
        h_tilde = torch.cat([h_node_feats, h_agg], dim=1)
        h_tilde = self.phi_h_node(h_tilde)

        return h_node_feats
        


class VecNet(GNNBase):
    def __init__(self, hparams) -> None:
        super().__init__(hparams)

        self.n_equi_hidden = self.hparams["n_hidden"]
        self.n_free_hidden = int(self.hparams["n_hidden"] * self.hparams["free_hidden_ratio"])
        self.n_layers = self.hparams["n_layers"]
        self.n_graph_iters = self.hparams["n_graph_iters"]
        self.n_output = self.hparams["n_output"]
        self.c_weight = self.hparams["c_weight"]
        self.n_input = self.hparams["n_input"]
        if self.hparams["equi_output"]:
            self.n_output = 3*self.n_equi_hidden
        else:
            self.n_output = self.n_equi_hidden + 2*self.hparams["vector_dim"] + 2*self.n_free_hidden

        input_edge_attributes, input_scalar_attributes = 4, 1+self.n_equi_hidden
        subsequent_edge_attributes, subsequent_scalar_attributes = 2+3*self.n_equi_hidden, 2*self.n_equi_hidden

        self.EBs = nn.ModuleList(
            [ EB(input_edge_attributes, input_scalar_attributes, n_equi_hidden=self.n_equi_hidden, n_free_hidden=self.n_free_hidden, self.n_layers, self.c_weight) ]
            +
            [
                EB(subsequent_edge_attributes, subsequent_scalar_attributes, n_equi_hidden=self.n_equi_hidden, n_free_hidden=self.n_free_hidden, nb_node_layer=self.n_layers, c_weight=self.c_weight)
                for _ in range(1, self.n_graph_iters)
            ]
        )


        # MLP to produce edge weights
        self.edge_mlp = make_mlp(
            self.n_output,
            [self.n_hidden] * self.n_layers + [1],
            layer_norm=True,
        )

    def forward(self, x, edge_index):

        v = x[:, :2]
        s = x[:, 2].unsqueeze(-1) # Think this is necessary...
        h = None
        e = None
        
        for i in range(self.n_graph_iters):                                                        
            v, s, h, e = self.EBs[i](v, edge_index, s, h, e)

        if self.hparams["equi_output"]:
            m = torch.cat([s[edge_index[1]], s[edge_index[0]], e], dim=1)
        else:
            m = torch.cat([v[edge_index[1]], v[edge_index[0]], h[edge_index[1]], h[edge_index[0]], e], dim=1)

        return self.edge_mlp(m)