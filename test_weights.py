import torch
from phormerModel import Graphormer
from configuration import Config
from zincdata import ZincDataset
import dgl

from trainpipeline import evaluate_test_loss


if __name__ == "__main__":
    cfg = Config()
    dataset = ZincDataset(cfg)
    # Dataloader for test samples
    test_loader = dgl.dataloading.GraphDataLoader(
        dataset.test_samples,
        batch_size=cfg.test_batch_size,
        shuffle=False,
        collate_fn=dataset.collate,
        num_workers=8,
    )
    weights_path = "model.pth"
    model = Graphormer(cfg)
    model.load_state_dict(torch.load(weights_path))
    avg_test_loss = evaluate_test_loss(model, test_loader)
    print(f"Average Test Loss (MAE): {avg_test_loss}")
