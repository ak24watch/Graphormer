from dataclasses import dataclass
import torch.nn as nn

@dataclass
class Config:
    d_model: int = 80
    d_ffn: int = 80
    d_head: int = 8
    n_heads: int = 10
    n_layers: int = 12  # predictive model layers
    ffn_activation = nn.GELU()
    ffn_dropout: float = 0.2
    attention_dropout: float = 0.2
    edge_encoding = False  # edge encoding

    K: int = 16  # number of eigenvectors and eigenvalues
    d_signet: int = 80
    phi_num_layers: int = 8
    rho_num_layers: int = 4
    max_eigen_value: int = -1
    eigenvalue: bool = False
    pos_emb: bool = True

    debug: bool = False
    add_pos_emb: bool = False
    deg_emb: bool = False
    regression_output_dim: int = 1
    num_bond_types: int = 4
    num_atom_types: int = 28
    max_num_nodes: int = 37
    max_degrees: int = 10
    max_path_length: int = 5
    max_dist: int = 8

    train_batch_size: int = 256 * 2
    valid_batch_size: int = 256
    test_batch_size: int = 256

    eps: float = 1e-8
    lr: float = 2e-3
    betas: tuple = (0.9, 0.999)
    weight_decay: float = 0
    num_train_samples: int = 10000
    num_valid_samples: int = 1000
    num_test_samples: int = 1000
    out_activation: nn.Module = nn.ReLU()

    node_classification: bool = True
    num_classes: int = 0
    epochs: int = 100
    num_node_features: int = 0