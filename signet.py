import einops
import torch.nn as nn
import torch
import torch.nn.functional as F


class Signet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        if cfg.eigenvalue:
            self.in_dim = 2
        else:
            self.in_dim = 1
        # phi parameters
        self.phi_layers = nn.ModuleList()
        for _ in range(cfg.phi_num_layers):
            self.phi_layers.append(
                nn.Linear(self.in_dim if _ == 0 else cfg.d_signet, cfg.d_signet)
            )

        # rho parameters
        self.rho_layers = nn.ModuleList()
        for _ in range(cfg.rho_num_layers):
            self.rho_layers.append(
                nn.Linear(
                    cfg.d_signet * cfg.K if _ == 0 else cfg.d_signet, cfg.d_signet
                )
            )

        self.out_pe = nn.Linear(cfg.d_signet, cfg.d_model)



    def forward(self, eigenvec, eigen_value):
        # eigenvec: [batch, max_num_nodes, K]
        eigenvec = einops.rearrange(
            eigenvec, "batch max_num_nodes K -> batch max_num_nodes K 1"
        )

        # eigen_value: [batch, max_num_nodes, K]
        if self.cfg.eigenvalue:
            eigen_value = einops.rearrange(
                eigen_value, "batch max_num_nodes K -> batch max_num_nodes K 1"
            )
            lap_pe = torch.cat(
                (eigenvec, eigen_value), dim=-1
            )  # [batch, max_num_nodes, K, 2]
        else:
            lap_pe = eigenvec # [batch, max_num_nodes, K, 1]

        pe = lap_pe # [batch, max_num_nodes, K, 2 or 1]
        for layer in self.phi_layers:
            pe = F.relu(layer(pe))

        # Rearranging to combine K and d_mlp
        pe = einops.rearrange(pe, "batch position K d_mlp -> batch position (K d_mlp)")

        for layer in self.rho_layers:
            pe = F.relu(layer(pe))

        pe = self.out_pe(pe)
        # print("pe is", pe)
        return pe  # [batch, max_num_nodes, d_model]
