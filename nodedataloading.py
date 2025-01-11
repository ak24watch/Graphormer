import dgl
import dgl.data
from dgl import LapPE


class NodeDataLoading:
    def __init__(self, cfg, data_name):
        if data_name == "Cora":
            dataset = dgl.data.CoraGraphDataset(verbose=False)
        elif data_name == "CiteSeer":
            dataset = dgl.data.CiteseerGraphDataset(verbose=False)
        elif data_name == "PubMed":
            dataset = dgl.data.PubmedGraphDataset(verbose=False)
        else:
            raise ValueError("Dataset not supported")
        cfg.num_classes = dataset.num_classes
      
        print(f"Number of classes: {cfg.num_classes}")
        self.graph = dataset[0]

        cfg.num_node_features = self.graph.ndata["feat"].shape[-1]

        if cfg.eigenvalue:
            lap_pe_transform = LapPE(cfg.K, "eigenvec", "eigen_value")
        else:
            lap_pe_transform = LapPE(cfg.K, "eigenvec")

        self.graph.ndata["spd"] = dgl.shortest_dist(self.graph)

        lap_pe_transform(self.graph)
        self.cfg = cfg

    def collate(self):
        nfeat = self.graph.ndata["feat"]
        nfeat = nfeat.unsqueeze(0)
        if self.cfg.eigenvalue:
            eigen_value = self.graph.ndata["eigen_value"]
            eigen_value = eigen_value.unsqueeze(0)
        eigenvec = self.graph.ndata["eigenvec"]
        eigenvec = eigenvec.unsqueeze(0)
        spd = self.graph.ndata["spd"]
        spd = spd.unsqueeze(0)
        labels = self.graph.ndata["label"]
        labels = labels.unsqueeze(0)
        train_mask = self.graph.ndata["train_mask"]
        train_mask = train_mask.unsqueeze(0)
        test_mask = self.graph.ndata["test_mask"]
        test_mask = test_mask.unsqueeze(0)
        valid_mask = self.graph.ndata["val_mask"]
        valid_mask = valid_mask.unsqueeze(0)

        if self.cfg.deg_emb:
            in_degree = self.graph.in_degrees() + 1
            out_degree = self.graph.out_degrees() + 1
            in_degree = in_degree.unsqueeze(0)
            out_degree = out_degree.unsqueeze(0)

        return (
            in_degree if self.cfg.deg_emb else None,
            out_degree if self.cfg.deg_emb else None,
            nfeat,
            eigenvec,
            eigen_value if self.cfg.eigenvalue else None,
            spd,
            labels,
            train_mask,
            valid_mask,
            test_mask,
        )
