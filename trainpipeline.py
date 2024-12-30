from zincdata import ZincDataset
import dgl
from phormerModel import Graphormer
import torch

def train_epoch(model, optimizer, data_loader):
    """
    Train the model for one epoch.

    Args:
        model: The model to be trained.
        optimizer: The optimizer used for training.
        data_loader: The data loader providing the training data.

    Returns:
        The average loss for the epoch.
    """
    model.train()
    epoch_loss = 0

    for (
        batch_labels,
        attn_mask,
        node_feat,
        in_degree,
        out_degree,
        path_data,
        dist,
    ) in data_loader:
        optimizer.zero_grad()
        batch_scores = model(
            node_feat,
            in_degree,
            out_degree,
            path_data,
            dist,
            attn_mask=attn_mask,
        )
        absolute_error = torch.abs(batch_scores - batch_labels)
        loss = torch.mean(absolute_error)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # Gradient clipping
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss /= len(data_loader)
    return epoch_loss

def evaluate_network(model, data_loader):
    """
    Evaluate the model on the validation data.

    Args:
        model: The model to be evaluated.
        data_loader: The data loader providing the validation data.

    Returns:
        The average loss for the validation data.
    """
    model.eval()
    epoch_loss = 0
    for (
        batch_labels,
        attn_mask,
        node_feat,
        in_degree,
        out_degree,
        path_data,
        dist,
    ) in data_loader:
        batch_scores = model(
            node_feat,
            in_degree,
            out_degree,
            path_data,
            dist,
            attn_mask=attn_mask,
        )
        absolute_error = torch.abs(batch_scores - batch_labels)
        loss = torch.mean(absolute_error)
        epoch_loss += loss.item()
    epoch_loss /= len(data_loader)
    return epoch_loss

def train_val_pipeline():
    """
    Train and validate the model.

    This function initializes the dataset, creates data loaders for training and validation,
    initializes the model and optimizer, and trains the model for a specified number of epochs.
    """
    dataset = ZincDataset()

    print(
        f"train, test, val sizes: {len(dataset.train)}, "
        f"{len(dataset.test)}, {len(dataset.val)}."
    )

    train_loader = dgl.dataloading.GraphDataLoader(
        dataset=dataset.train,
        collate_fn=dataset.collate,
        batch_size=256,  # Updated batch size
        shuffle=True,
    )

    val_loader = dgl.dataloading.GraphDataLoader(
        dataset.val,
        batch_size=256,  # Updated batch size
        shuffle=False,
        collate_fn=dataset.collate,
    )

    model = Graphormer(
        num_atoms=dataset.max_num_nodes,
        max_in_degree=dataset.max_in_degree,
        max_out_degree=dataset.max_out_degree,
        num_spatial=dataset.max_dist,
        multi_hop_max_dist=dataset.max_dist,
        num_encoder_layers=12,
        embedding_dim=80,
        ffn_embedding_dim=80,
        num_attention_heads=8,
        dropout=0.1,
    )
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=2e-4,  # Peak learning rate
        eps=1e-8, 
        weight_decay=0.01, 
        betas=(0.9, 0.999)
    )

    for epoch in range(10000):  # Max epochs
        epoch_train_loss = train_epoch(
            model,
            optimizer,
            train_loader,
        )
        epoch_val_loss = evaluate_network(model, val_loader)

        print(
            f"Epoch={epoch + 1} | train_loss={epoch_train_loss:.3f} | val_loss={epoch_val_loss:.3f}"
        )

if __name__ == "__main__":
    train_val_pipeline()
