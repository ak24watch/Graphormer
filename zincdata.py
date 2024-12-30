import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import dgl
import dgl.data

class ZincDataset(torch.utils.data.Dataset):
    """
    ZincDataset class for loading and processing the ZINC dataset.

    This class handles the loading of the ZINC dataset, creating train, validation,
    and test splits, and processing the graphs and labels.

    Attributes:
        train (list): List of training samples.
        val (list): List of validation samples.
        test (list): List of test samples.
        max_dist (int): Maximum shortest path distance in the dataset.
        max_in_degree (int): Maximum in-degree in the dataset.
        max_out_degree (int): Maximum out-degree in the dataset.
        max_num_nodes (int): Maximum number of nodes in the dataset.
    """
    def __init__(self):
        train_dataset = dgl.data.ZINCDataset(mode="train")[:256*14]
        valid_dataset = dgl.data.ZINCDataset(mode="valid")[:256*2]
        test_dataset = dgl.data.ZINCDataset(mode="test")[:256*2]

        train_samples = [
            (graph, label) for graph, label in zip(train_dataset[0], train_dataset[1])
        ]
        valid_samples = [
            (graph, label) for graph, label in zip(valid_dataset[0], valid_dataset[1])
        ]
        test_samples = [
            (graph, label) for graph, label in zip(test_dataset[0], test_dataset[1])
        ]

        self.train = train_samples
        self.val = valid_samples
        self.test = test_samples
        self.max_dist = 0
        self.max_in_degree = 0
        self.max_out_degree = 0
        self.max_num_nodes = 0

        for dataset in [train_samples, valid_samples, test_samples]:
            for g, labels in dataset:
                spd, path = dgl.shortest_dist(g, return_paths=True)
                g.ndata["spd"] = spd
                g.ndata["path"] = path
                dist_maxi = torch.max(spd).item()
                if dist_maxi > self.max_dist:
                    self.max_dist = dist_maxi
                in_degree_maxi = torch.max(g.in_degrees()).item()
                if in_degree_maxi > self.max_in_degree:
                    self.max_in_degree = in_degree_maxi
                out_degree_maxi = torch.max(g.out_degrees()).item()
                if out_degree_maxi > self.max_out_degree:
                    self.max_out_degree = out_degree_maxi
                max_nodes = g.num_nodes()
                if max_nodes > self.max_num_nodes:
                    self.max_num_nodes = max_nodes

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

        attn_mask = torch.zeros(num_graphs, max_num_nodes + 1, max_num_nodes + 1)

        node_feat = []
        in_degree, out_degree = [], []
        path_data = []

        dist = -torch.ones((num_graphs, max_num_nodes, max_num_nodes), dtype=torch.long)

        for i in range(num_graphs):
            attn_mask[i, :, num_nodes[i] + 1 :] = 1
            attn_mask[i, num_nodes[i] + 1 :, :] = 1

            nd_feat = graphs[i].ndata["feat"] + 1
            if len(nd_feat.shape) == 1:
                nd_feat = nd_feat.unsqueeze(1)
            node_feat.append(nd_feat)

            in_degree.append(
                torch.clamp(graphs[i].in_degrees() + 1, min=0, max=self.max_in_degree)
            )
            out_degree.append(
                torch.clamp(graphs[i].out_degrees() + 1, min=0, max=self.max_out_degree)
            )

            path = graphs[i].ndata["path"]
            path_len = path.size(dim=2)
            max_len = self.max_dist
            if (path_len >= max_len):
                shortest_path = path[:, :, :max_len]
            else:
                p1d = (0, max_len - path_len)
                shortest_path = F.pad(path, p1d, "constant", -1)
            pad_num_nodes = max_num_nodes - num_nodes[i]
            p3d = (0, 0, 0, pad_num_nodes, 0, pad_num_nodes)
            shortest_path = F.pad(shortest_path, p3d, "constant", -1)

            edata = graphs[i].edata["feat"] + 1
            if len(edata.shape) == 1:
                edata = edata.unsqueeze(-1)
            edata = torch.cat((edata, torch.zeros(1, edata.shape[1])), dim=0)
            path_data.append(edata[shortest_path])

            dist[i, : num_nodes[i], : num_nodes[i]] = graphs[i].ndata["spd"]

        node_feat = pad_sequence(node_feat, batch_first=True)
        in_degree = pad_sequence(in_degree, batch_first=True)
        out_degree = pad_sequence(out_degree, batch_first=True)

        return (
            torch.stack(labels).reshape(num_graphs, -1),
            attn_mask,
            node_feat,
            in_degree,
            out_degree,
            torch.stack(path_data),
            dist,
        )
