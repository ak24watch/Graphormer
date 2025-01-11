import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import dgl
import dgl.data
from dgl import LapPE

class ZincDataset(torch.utils.data.Dataset):
    """
    ZincDataset class for loading and processing the ZINC dataset.

    This class handles the loading of the ZINC dataset, creating train, validation,
    and test splits, and processing the graphs and labels.

    Attributes:
        train_samples (list): List of training samples.
        valid_samples (list): List of validation samples.
        test_samples (list): List of test samples.
        cfg (object): Configuration object containing dataset parameters.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        if cfg.eigenvalue:
            lap_pe_transform = LapPE(cfg.K, "eigenvec", "eigen_value")
        else:
            lap_pe_transform = LapPE(cfg.K, "eigenvec")
        train_dataset = dgl.data.ZINCDataset(mode="train")
        valid_dataset = dgl.data.ZINCDataset(mode="valid")
        test_dataset = dgl.data.ZINCDataset(mode="test")
        train_indices = torch.randperm(len(train_dataset))
        valid_indices = torch.randperm(len(valid_dataset))
        test_indices = torch.randperm(len(test_dataset))

        self.train_samples = [
            train_dataset[i] for i in train_indices[: cfg.num_train_samples]
        ]
        self.valid_samples = [
            valid_dataset[i] for i in valid_indices[: cfg.num_valid_samples]
        ]
        self.test_samples = [
            test_dataset[i] for i in test_indices[: cfg.num_test_samples]
        ]

        for graph_set in [self.train_samples, self.valid_samples, self.test_samples]:
            for graph, _ in graph_set:
                if cfg.edge_encoding:
                    graph.ndata["spd"], graph.ndata["path"] = dgl.shortest_dist(
                        graph, return_paths=True
                    )
                else:
                    graph.ndata["spd"] = dgl.shortest_dist(graph)

                lap_pe_transform(graph)

                graph.ndata["feat"] = graph.ndata["feat"].to(torch.long)
                graph.edata["feat"] = graph.edata["feat"].to(torch.long)

    def collate(self, samples):
        """
        Custom collate function to batch graphs, labels, and additional data.

        Args:
            samples (list): List of samples, where each sample is a tuple (graph, label).

        Returns:
            tuple: Batched data including labels, attention mask, node features,
                   in-degrees, out-degrees, path data, and distances.
        """
        graphs, labels = zip(*samples)
        num_graphs = len(graphs)
        num_nodes = [g.num_nodes() for g in graphs]
        max_num_nodes = max(num_nodes)

        attn_mask = torch.ones(num_graphs, max_num_nodes + 1, max_num_nodes + 1)

        node_feat_list = []
        if self.cfg.deg_emb:
            in_degree_list, out_degree_list = [], []
        if self.cfg.edge_encoding:
            path_edata_list = []
        eigen_vecs_list = []
        if self.cfg.eigenvalue:
            eigen_value_list = []

        dist = -torch.ones((num_graphs, max_num_nodes, max_num_nodes), dtype=torch.long)

        for i in range(num_graphs):
            attn_mask[i, : num_nodes[i] + 1, : num_nodes[i] + 1] = 0
            node_feat = graphs[i].ndata["feat"] + 1
            if len(node_feat.shape) == 1:
                node_feat = node_feat.unsqueeze(1)
            node_feat_list.append(node_feat)

            if self.cfg.deg_emb:
                in_degree_list.append(graphs[i].in_degrees() + 1)
                out_degree_list.append(graphs[i].out_degrees() + 1)

            if self.cfg.edge_encoding:
                path = graphs[i].ndata["path"]
                path_len = path.size(dim=2)

                if path_len >= self.cfg.max_path_length:
                    path = path[:, :, : self.cfg.max_path_length]
                else:
                    p1d = (0, self.cfg.max_path_length - path_len)
                    path = F.pad(path, p1d, "constant", -1)
                pad_num_nodes = max_num_nodes - num_nodes[i]
                p3d = (0, 0, 0, pad_num_nodes, 0, pad_num_nodes)
                path = F.pad(path, p3d, "constant", -1)

                edata = graphs[i].edata["feat"] + 1
                if len(edata.shape) == 1:
                    edata = edata.unsqueeze(-1)
                edata = torch.cat(
                    (edata, torch.zeros((1, edata.shape[1]), dtype=torch.long)), dim=0
                )
                path_edata = edata[
                    path
                ]
                path_edata_list.append(path_edata)

            dist[i, : num_nodes[i], : num_nodes[i]] = graphs[i].ndata["spd"]

            if self.cfg.eigenvalue:
                eigen, eigen_value = (
                    graphs[i].ndata["eigenvec"],
                    graphs[i].ndata["eigen_value"],
                )
                eigen_vecs_list.append(eigen)
                eigen_value_list.append(eigen_value)

            else:
                eigen = graphs[i].ndata["eigenvec"]
                eigen_vecs_list.append(eigen)

        batched_node_feat = pad_sequence(node_feat_list, batch_first=True)
        if self.cfg.deg_emb:
            batched_indegree = pad_sequence(in_degree_list, batch_first=True)
            batched_outdegree = pad_sequence(out_degree_list, batch_first=True)

        batched_eigen_vecs = pad_sequence(eigen_vecs_list, batch_first=True)
        if self.cfg.eigenvalue:
            batched_eigen_value = pad_sequence(
                eigen_value_list, batch_first=True
            )

        return (
            torch.stack(labels).reshape(num_graphs, -1),
            attn_mask,
            batched_node_feat,
            batched_eigen_vecs,
            batched_eigen_value if self.cfg.eigenvalue else None,
            batched_indegree if self.cfg.deg_emb else None,
            batched_outdegree if self.cfg.deg_emb else None,
            torch.stack(path_edata_list) if self.cfg.edge_encoding else None,
            dist,
        )
