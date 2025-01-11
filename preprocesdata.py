import torch
import pickle
import dgl
import os
from dataclasses import dataclass
from zincdata import ZincDataset
@dataclass
class Config:
    d_model: int = 80
    d_ffn: int = 160
    d_head: int = 8
    n_heads: int = 10
    n_layers: int = 8  # predictive model layers
    ffn_activation = nn.GELU()
    ffn_dropout: float = 0.1
    attention_dropout: float = 0.1

    K: int = 6  # number of eigenvectors and eigenvalues
    d_signet: int = 40
    phi_num_layers: int = 2
    rho_num_layers: int = 2
    max_eigen_value: int = -1
    eigenvalue: bool = False

    debug: bool = False
    add_pos_emb: bool = True
    deg_emb: bool = False
    regression_output_dim: int = 1
    num_bond_types: int = 4
    num_atom_types: int = 28
    max_num_nodes: int = 37
    max_degrees: int = 10
    max_path_length: int = 5
    max_dist: int = 8

    train_batch_size: int = 100
    valid_batch_size: int = 20
    test_batch_size: int = 20

    eps: float = 1e-8
    lr: float = 2e-3
    betas: tuple = (0.9, 0.999)
    weight_decay: float = 0.01
    num_train_samples: int = 1000
    num_valid_samples: int = 100
    num_test_samples: int = 100



def save_dataset(dataset, filename):
    with open(filename, "wb") as f:
        pickle.dump(dataset, f)


    # Save the training and validation samples
    save_dataset(dataset.train_samples, "train_samples.pkl")
    save_dataset(dataset.valid_samples, "valid_samples.pkl")
    cfg = Config()
        dataset = ZincDataset(cfg=cfg)

        # Check if precomputed datasets exist
        if os.path.exists('train_samples.pkl') and os.path.exists('valid_samples.pkl'):
            print("Loading precomputed datasets...")
            train_samples = load_dataset('train_samples.pkl')
            valid_samples = load_dataset('valid_samples.pkl')
        else:
            print("Precomputed datasets not found. Computing and saving them...")
            train_samples = dataset.train_samples
            valid_samples = dataset.valid_samples
            save_dataset(train_samples, 'train_samples.pkl')
            save_dataset(valid_samples, 'valid_samples.pkl')

        print(f"Number of training samples: {len(train_samples)}")
        print(f"Number of validation samples: {len(valid_samples)}")
        print(f"Number of test samples: {len(dataset.test_samples)}")

        train_loader = dgl.dataloading.GraphDataLoader(
            dataset=train_samples,
            collate_fn=dataset.collate,
            batch_size=cfg.train_batch_size,
            shuffle=True,
            num_workers=4,
        )

        valid_loader = dgl.dataloading.GraphDataLoader(
            dataset=valid_samples,
            batch_size=cfg.valid_batch_size,
            shuffle=False,
            collate_fn=dataset.collate,
            num_workers=4,
        )
