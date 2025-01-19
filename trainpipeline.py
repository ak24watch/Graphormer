from zincdata import ZincDataset
import dgl
from phormerModel import Graphormer
import torch
from torchsummary import summary

from configuration import Config
from tqdm import tqdm
import plotly.graph_objects as go

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
        eigen_vecs,
        eigen_values,
        in_degree,
        out_degree,
        path_data,
        dist,
    ) in tqdm(data_loader, desc="Training"):
        optimizer.zero_grad()
        batch_scores = model(
            node_feat,
            in_degree,
            out_degree,
            path_data,
            dist,
            eigen_vecs,
            eigen_values,
            attn_mask=attn_mask,
        )
        absolute_error = torch.abs(batch_scores - batch_labels)
        loss = torch.mean(absolute_error)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
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
        eigen_vecs,
        eigen_values,
        in_degree,
        out_degree,
        path_data,
        dist,
    ) in tqdm(data_loader, desc="evaluate"):
        batch_scores = model(
            node_feat,
            in_degree,
            out_degree,
            path_data,
            dist,
            eigen_vecs,
            eigen_values,
            attn_mask=attn_mask,
        )
        absolute_error = torch.abs(batch_scores - batch_labels)
        loss = torch.mean(absolute_error)
        epoch_loss += loss.item()
    epoch_loss /= len(data_loader)
    return epoch_loss

def evaluate_test_loss(model, data_loader):
    """
    Evaluate the model on the test data.

    Args:
        model: The model to be evaluated.
        data_loader: The data loader providing the test data.

    Returns:
        The average loss for the test data.
    """
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for (
            batch_labels,
            attn_mask,
            node_feat,
            eigen_vecs,
            eigen_values,
            in_degree,
            out_degree,
            path_data,
            dist,
        ) in tqdm(data_loader, desc="Testing"):
            batch_scores = model(
                node_feat,
                in_degree,
                out_degree,
                path_data,
                dist,
                eigen_vecs,
                eigen_values,
                attn_mask=attn_mask,
            )
            absolute_error = torch.abs(batch_scores - batch_labels)
            loss = torch.mean(absolute_error)
            epoch_loss += loss.item()
    epoch_loss /= len(data_loader)
    return epoch_loss

def plot_losses(train_losses, val_losses):
    """
    Plot training and validation losses.

    Args:
        train_losses: List of training losses.
        val_losses: List of validation losses.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=train_losses, mode='lines', name='Training Loss'))
    fig.add_trace(go.Scatter(y=val_losses, mode='lines', name='Validation Loss'))
    fig.update_layout(title='Training and Validation Loss', xaxis_title='Epoch', yaxis_title='Loss')
    fig.write_image("losses_plot.png")

def train_val_pipeline():
    """
    Train and validate the model.

    This function initializes the dataset, creates data loaders for training and validation,
    initializes the model and optimizer, and trains the model for a specified number of epochs.
    """
    cfg = Config()
    dataset = ZincDataset(cfg=cfg)

    print(f"Number of training samples: {len(dataset.train_samples)}")
    print(f"Number of validation samples: {len(dataset.valid_samples)}")
    print(f"Number of test samples: {len(dataset.test_samples)}")

    train_loader = dgl.dataloading.GraphDataLoader(
        dataset=dataset.train_samples,
        collate_fn=dataset.collate,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )

    valid_loader = dgl.dataloading.GraphDataLoader(
        dataset.valid_samples,
        batch_size=cfg.valid_batch_size,
        shuffle=False,
        collate_fn=dataset.collate,
        num_workers=cfg.num_workers,
    )

    test_loader = dgl.dataloading.GraphDataLoader(
        dataset.test_samples,
        batch_size=cfg.test_batch_size,
        shuffle=False,
        collate_fn=dataset.collate,
        num_workers=cfg.num_workers,
    )

    model = Graphormer(cfg)
    summary(model)
  

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        eps=cfg.eps,
        weight_decay=cfg.weight_decay,
        betas=cfg.betas,
    )

    train_losses = []
    val_losses = []

    for epoch in range(300):
        epoch_train_loss = train_epoch(
            model,
            optimizer,
            train_loader,
        )
        epoch_val_loss = evaluate_network(model, valid_loader)

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

        print(
            f"Epoch={epoch + 1} | train_loss={epoch_train_loss:.3f} | val_loss={epoch_val_loss:.3f}"
        )
    torch.save(model.state_dict(), "model.pth")
    test_loss = evaluate_test_loss(model, test_loader)

    plot_losses(train_losses, val_losses)

    print(f"Test Loss: {test_loss:.3f}")

if __name__ == "__main__":
    
    train_val_pipeline()
