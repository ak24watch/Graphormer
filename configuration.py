from dataclasses import dataclass
import torch.nn as nn

@dataclass
class Config:
    """
    Configuration class for setting model parameters.

    Attributes:
        d_model (int): Dimension of the model.
        d_ffn (int): Dimension of the feed-forward network.
        d_head (int): Dimension of each attention head.
        n_heads (int): Number of attention heads.
        n_layers (int): Number of encoder layers.
        ffn_activation (nn.Module): Activation function for the feed-forward network.
        ffn_dropout (float): Dropout rate for the feed-forward network.
        attention_dropout (float): Dropout rate for the attention mechanism.
        edge_encoding (bool): Whether to use edge encoding.
        K (int): Number of eigenvectors and eigenvalues.
        d_signet (int): Dimension of the Signet model.
        phi_num_layers (int): Number of layers in the phi network.
        rho_num_layers (int): Number of layers in the rho network.
        max_eigen_value (int): Maximum eigenvalue.
        eigenvalue (bool): Whether to use eigenvalues.
        pos_emb (bool): Whether to use positional embeddings.
        debug (bool): Debug mode.
        add_pos_emb (bool): Whether to add positional embeddings.
        deg_emb (bool): Whether to use degree embeddings.
        regression_output_dim (int): Dimension of the regression output.
        num_bond_types (int): Number of bond types.
        num_atom_types (int): Number of atom types.
        max_num_nodes (int): Maximum number of nodes.
        max_degrees (int): Maximum degrees.
        max_path_length (int): Maximum path length.
        max_dist (int): Maximum distance.
        train_batch_size (int): Batch size for training.
        valid_batch_size (int): Batch size for validation.
        test_batch_size (int): Batch size for testing.
        eps (float): Epsilon value for the optimizer.
        lr (float): Learning rate.
        betas (tuple): Betas for the optimizer.
        weight_decay (float): Weight decay for the optimizer.
        num_train_samples (int): Number of training samples.
        num_valid_samples (int): Number of validation samples.
        num_test_samples (int): Number of test samples.
        out_activation (nn.Module): Activation function for the output layer.
    """
    d_model: int = 80
    d_ffn: int = 80
    d_head: int = 8
    n_heads: int = 10
    n_layers: int = 12
    ffn_activation = nn.GELU()
    ffn_dropout: float = 0.2
    attention_dropout: float = 0.2
    edge_encoding = True

    K: int = 10
    d_signet: int = 80
    phi_num_layers: int = 2
    rho_num_layers: int = 2
    max_eigen_value: int = -1
    eigenvalue: bool = False
    pos_emb: bool = True

  
    add_pos_emb: bool = False
    deg_emb: bool = True
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
    num_train_samples: int = 100
    num_valid_samples: int = 100
    num_test_samples: int = 100
    out_activation: nn.Module = nn.ReLU()

