from zincdata import ZincDataset
import dgl
from phormerModel import Graphormer
import torch
import torch.nn as nn
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

def evaluate_test_accuracy(model, data_loader):
    """
    Evaluate the model on the test data.

    Args:
        model: The model to be evaluated.
        data_loader: The data loader providing the test data.

    Returns:
        The average accuracy for the test data.
    """
    model.eval()
    correct = 0
    total = 0
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
            predicted = torch.round(batch_scores)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
    return correct / total

def plot_accuracies(train_accuracies, val_accuracies, test_accuracy):
    """
    Plot training, validation, and test accuracies.

    Args:
        train_accuracies: List of training accuracies.
        val_accuracies: List of validation accuracies.
        test_accuracy: Test accuracy.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=train_accuracies, mode='lines', name='Training Accuracy'))
    fig.add_trace(go.Scatter(y=val_accuracies, mode='lines', name='Validation Accuracy'))
    fig.add_trace(go.Scatter(y=[test_accuracy] * len(train_accuracies), mode='lines', name='Test Accuracy'))
    fig.update_layout(title='Training, Validation, and Test Accuracy', xaxis_title='Epoch', yaxis_title='Accuracy')
    fig.write_image("accuracies_plot.png")

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
        num_workers=0,
    )

    valid_loader = dgl.dataloading.GraphDataLoader(
        dataset.valid_samples,
        batch_size=cfg.valid_batch_size,
        shuffle=False,
        collate_fn=dataset.collate,
        num_workers=0,
    )

    test_loader = dgl.dataloading.GraphDataLoader(
        dataset.test_samples,
        batch_size=cfg.test_batch_size,
        shuffle=False,
        collate_fn=dataset.collate,
        num_workers=0,
    )

    model = Graphormer(cfg)
    model = nn.DataParallel(model)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        eps=cfg.eps,
        weight_decay=cfg.weight_decay,
        betas=cfg.betas,
    )

    train_accuracies = []
    val_accuracies = []

    for epoch in range(300):
        epoch_train_loss = train_epoch(
            model,
            optimizer,
            train_loader,
        )
        epoch_val_loss = evaluate_network(model, valid_loader)

        train_accuracies.append(1 - epoch_train_loss)
        val_accuracies.append(1 - epoch_val_loss)

        print(
            f"Epoch={epoch + 1} | train_loss={epoch_train_loss:.3f} | val_loss={epoch_val_loss:.3f}"
        )

    test_accuracy = evaluate_test_accuracy(model, test_loader)
    plot_accuracies(train_accuracies, val_accuracies, test_accuracy)

    print(f"Test Accuracy: {test_accuracy:.3f}")

    torch.save(model.state_dict(), "model.pth")

if __name__ == "__main__":
    train_val_pipeline()
