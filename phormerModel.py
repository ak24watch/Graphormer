import torch
import torch.nn as nn
from centralityencoding import CentralityEncoder
from spaceencoding import SpatialEncoder
from edgeencoding import EdgeEncoder
from encoder import Encoder
from signet import Signet
import einops


class Graphormer(nn.Module):
    """
    Graphormer model for graph representation learning.

    Args:
        cfg (object): Configuration object containing model parameters.
    """

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        if cfg.pos_emb:
            self.signet = Signet(cfg)
        self.atom_encoder = nn.Embedding(
            cfg.num_atom_types + 1, cfg.d_model, padding_idx=0, device=cfg.device
        )
        if cfg.edge_encoding:
            self.bond_encoder = nn.Embedding(
                cfg.num_bond_types + 1, cfg.d_model, padding_idx=0, device=cfg.device
            )
        self.graph_node_enoceder = nn.Embedding(
            1, 2 * cfg.d_model if cfg.concat_pos_emb else cfg.d_model, device=cfg.device
        )
        if cfg.deg_emb:
            self.degree_encoder = CentralityEncoder(cfg)
        if cfg.edge_encoding:
            self.path_encoder = EdgeEncoder(cfg)
        self.spatial_encoder = SpatialEncoder(cfg)
        self.graph_node_virtual_distance_encoder = nn.Embedding(
            1, cfg.n_heads, device=cfg.device
        )
        self.emb_layer_norm = nn.LayerNorm(
            2 * cfg.d_model if cfg.concat_pos_emb else cfg.d_model, device=cfg.device
        )

        self.layers = nn.ModuleList([Encoder(cfg) for _ in range(cfg.n_layers)])
        self.lm_head_transform_weight = nn.Linear(
            2 * cfg.d_model if cfg.concat_pos_emb else cfg.d_model,
            2 * cfg.d_model if cfg.concat_pos_emb else cfg.d_model,
            bias=False,
            device=cfg.device,
        )
        self.layer_norm = nn.LayerNorm(
            2 * cfg.d_model if cfg.concat_pos_emb else cfg.d_model,
            device=cfg.device,
        )

        self.embed_out = nn.Linear(
            2 * cfg.d_model if cfg.concat_pos_emb else cfg.d_model,
            cfg.regression_output_dim,
            bias=False,
            device=cfg.device,
        )
        self.lm_output_learned_bias = nn.Parameter(
            torch.zeros((cfg.regression_output_dim), device=cfg.device)
        )

    def reset_output_layer_parameters(self):
        """
        Reset the parameters of the output layer.
        """
        self.lm_output_learned_bias = nn.Parameter(
            torch.zeros(1, device=self.cfg.device)
        )
        self.embed_out.reset_parameters()

    def forward(
        self,
        node_feat,
        in_degree,
        out_degree,
        path_data,
        dist,
        eigenvecs,
        eigen_value,
        attn_mask=None,
    ):
        """
        Forward pass for the Graphormer model.

        Args:
            node_feat (Tensor): Node feature tensor.
            in_degree (Tensor): In-degree tensor.
            out_degree (Tensor): Out-degree tensor.
            path_data (Tensor): Path data tensor.
            dist (Tensor): Distance tensor.
            eigenvecs (Tensor): Eigenvector tensor.
            eigen_value (Tensor): Eigenvalue tensor.
            attn_mask (Tensor, optional): Attention mask tensor.

        Returns:
            Tensor: Graph representation tensor.
        """
        num_graphs, max_num_nodes, _ = node_feat.shape

        node_emb = self.atom_encoder(node_feat)
        node_emb = einops.reduce(
            node_emb,
            "batch pos node_type d_model -> batch pos d_model",
            "mean",
        )
        if self.cfg.edge_encoding:
            path_edata_emb = self.bond_encoder(path_data)
            path_edata_emb = einops.reduce(
                path_edata_emb,
                "batch pos1 pos2 path_len edge_type_dim edge_dim -> batch pos1 pos2 path_len edge_dim",
                "mean",
            )

        if self.cfg.deg_emb:
            deg_emb = self.degree_encoder(in_degree, out_degree)
            node_emb += deg_emb

        if self.cfg.pos_emb:
            pos_emb = self.signet(eigenvecs, eigen_value)
            if self.cfg.concat_pos_emb:
                node_emb = torch.cat((pos_emb, node_emb), dim=-1)
            else:
                node_emb += pos_emb
                
        spatial_encoding = self.spatial_encoder(dist)
        if self.cfg.edge_encoding:
            path_encoding = self.path_encoder(dist, path_edata_emb)

        attn_bias = torch.zeros(
            num_graphs,
            max_num_nodes + 1,
            max_num_nodes + 1,
            self.cfg.n_heads,
            device=self.cfg.device,
        )
        attn_bias[:, 1:, 1:, :] = (
            path_encoding if self.cfg.edge_encoding else 0
        ) + spatial_encoding

        graph_node_saptial_bias = (
            self.graph_node_virtual_distance_encoder.weight.reshape(
                1, 1, self.cfg.n_heads
            )
        )
        attn_bias[:, 1:, 0, :] = attn_bias[:, 1:, 0, :] + graph_node_saptial_bias
        attn_bias[:, 0, :, :] = attn_bias[:, 0, :, :] + graph_node_saptial_bias
        graph_node_emb = einops.repeat(
            self.graph_node_enoceder.weight,
            "1 d_model -> num_graphs 1 d_model",
            num_graphs=num_graphs,
        )

        x = torch.cat((graph_node_emb, node_emb), dim=1)

        x = self.emb_layer_norm(x)
        for layer in self.layers:
            x = layer(
                x,
                att_mask=attn_mask,
                att_bias=attn_bias,
            )
        graph_rep = x[:, 0, :]
        graph_rep = self.layer_norm(
            self.cfg.out_activation(self.lm_head_transform_weight(graph_rep))
        )
        graph_rep = self.embed_out(graph_rep) + self.lm_output_learned_bias

        return graph_rep
