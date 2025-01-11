from nodedataloading import NodeDataLoading
import torch
import torch.nn.functional as F
from phormerModel import Graphormer
from configuration import Config
from tqdm import tqdm


def training(
    in_degree,
    out_degree,
    n_feat,
    eigenvec,
    eigen_value,
    spd,
    labels,
    train_mask,
    model,
    optimizer,
):
    """
    Perform a single training step.

    Args:
        in_degree: Input degree tensor.
        out_degree: Output degree tensor.
        n_feat: Node features tensor.
        eigenvec: Eigenvectors tensor.
        eigen_value: Eigenvalues tensor.
        spd: Shortest path distance tensor.
        labels: Ground truth labels tensor.
        train_mask: Mask tensor for training nodes.
        model: The model to train.
        optimizer: The optimizer for training.

    Returns:
        loss.item(): The training loss.
    """
    model.train()
    optimizer.zero_grad()
    output = model(n_feat, in_degree, out_degree, None, spd, eigenvec, eigen_value)
    loss = F.cross_entropy(output[train_mask], labels[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate(
    in_degree,
    out_degree,
    n_feat,
    eigenvec,
    eigen_value,
    spd,
    labels,
    mask,
    model,
):
    """
    Evaluate the model.

    Args:
        in_degree: Input degree tensor.
        out_degree: Output degree tensor.
        n_feat: Node features tensor.
        eigenvec: Eigenvectors tensor.
        eigen_value: Eigenvalues tensor.
        spd: Shortest path distance tensor.
        labels: Ground truth labels tensor.
        mask: Mask tensor for evaluation nodes.
        model: The model to evaluate.

    Returns:
        acc: The accuracy of the model on the given mask.
    """
    model.eval()
    with torch.no_grad():
        output = model(n_feat, in_degree, out_degree, None, spd, eigenvec, eigen_value)
    pred = output[mask].max(1)[1]
    acc = pred.eq(labels[mask]).sum().item() / mask.sum().item()
    return acc


def trainPipelineNodeClassify(dataset, cfg, optimizer, model):
    """
    Train and evaluate the model for node classification.

    Args:
        data_name: The name of the dataset.
        cfg: Configuration object.
        optimizer: The optimizer for training.
        model: The model to train and evaluate.
    """

    (
        in_degree,
        out_degree,
        nfeat,
        eigenvec,
        eigen_value,
        spd,
        labels,
        train_mask,
        valid_mask,
        test_mask,
    ) = dataset.collate()

    for epoch in tqdm(range(cfg.epochs), desc="Training Epochs"):
        train_loss = training(
            in_degree,
            out_degree,
            nfeat,
            eigenvec,
            eigen_value,
            spd,
            labels,
            train_mask,
            model,
            optimizer,
        )
        valid_acc = evaluate(
            in_degree,
            out_degree,
            nfeat,
            eigenvec,
            eigen_value,
            spd,
            labels,
            valid_mask,
            model,
        )
        test_acc = evaluate(
            in_degree,
            out_degree,
            nfeat,
            eigenvec,
            eigen_value,
            spd,
            labels,
            test_mask,
            model,
        )
        print(
            f"Epoch: {epoch + 1}, Train Loss: {train_loss}, "
            f"Validation Accuracy: {valid_acc}, Test Accuracy: {test_acc}"
        )


if __name__ == "__main__":
    cfg = Config()
    dataset = NodeDataLoading(cfg, "Cora")
    model = Graphormer(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    trainPipelineNodeClassify(dataset, cfg, optimizer, model)
    # trainPipelineNodeClassify("CiteSeer", cfg, optimizer, model)
    # trainPipelineNodeClassify("PubMed", cfg, optimizer, model)
