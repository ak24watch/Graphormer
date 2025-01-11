import einops
import torch.nn as nn
import torch
import torch.nn.functional as F

class Signet(nn.Module):
    """
    Signet model for encoding eigenvectors and eigenvalues.

    Args:
        cfg (object): Configuration object containing model parameters.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.in_dim = 2 if cfg.eigenvalue else 1

        self.phi_layers = nn.ModuleList()
        for _ in range(cfg.phi_num_layers):
            self.phi_layers.append(
                nn.Linear(self.in_dim if _ == 0 else cfg.d_signet, cfg.d_signet)
            )

        self.rho_layers = nn.ModuleList()
        for _ in range(cfg.rho_num_layers):
            self.rho_layers.append(
                nn.Linear(
                    cfg.d_signet * cfg.K if _ == 0 else cfg.d_signet, cfg.d_signet
                )
            )

        self.out_pe = nn.Linear(cfg.d_signet, cfg.d_model)

    def forward(self, eigenvec, eigen_value):
        """
        Forward pass for the Signet model.

        Args:
            eigenvec (Tensor): Eigenvector tensor.
            eigen_value (Tensor): Eigenvalue tensor.

        Returns:
            Tensor: Encoded positional embeddings.
        """
        eigenvec = einops.rearrange(
            eigenvec, "batch max_num_nodes K -> batch max_num_nodes K 1"
        )

        if self.cfg.eigenvalue:
            eigen_value = einops.rearrange(
                eigen_value, "batch max_num_nodes K -> batch max_num_nodes K 1"
            )
            lap_pe = torch.cat(
                (eigenvec, eigen_value), dim=-1
            )
        else:
            lap_pe = eigenvec

        pe = lap_pe
        for layer in self.phi_layers:
            pe = F.relu(layer(pe))

        pe = einops.rearrange(pe, "batch position K d_mlp -> batch position (K d_mlp)")

        for layer in self.rho_layers:
            pe = F.relu(layer(pe))

        pe = self.out_pe(pe)
        return pe
