# Graphormer

Graphormer is a graph representation learning model designed to encode various graph features, including node centrality, shortest path distances, and eigenvectors/eigenvalues. This repository contains the implementation of the Graphormer model and its components.

The model is based on the papers "Do Transformers Really Perform Bad for Graph Representation?" and "Sign and Basis Invariant Networks for Spectral Graph Representation Learning."

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Components](#components)
- [Training](#training)
- [Plotting](#plotting)
- [License](#license)

## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

## Usage

To use the Graphormer model, you need to prepare your dataset and configuration. Below is an example of how to initialize and use the model:

```python
from configuration import Config
from phormerModel import Graphormer

cfg = Config()
model = Graphormer(cfg)

# Example input data
node_feat = ...
in_degree = ...
out_degree = ...
path_data = ...
dist = ...
eigenvecs = ...
eigen_value = ...

output = model(node_feat, in_degree, out_degree, path_data, dist, eigenvecs, eigen_value)
```

## Configuration

The configuration for the Graphormer model is defined in the `Config` class in `configuration.py`. You can customize various parameters such as model dimensions, number of layers, dropout rates, and more.

```python
from configuration import Config

cfg = Config()
print(cfg)
```

## Components

### Spatial Encoder

Encodes shortest path distances using an embedding table.

### Signet

Encodes eigenvectors and eigenvalues using a series of linear layers.

### Centrality Encoder

Encodes node centrality features based on in-degrees and out-degrees.

### Edge Encoder

Encodes edge features along the shortest path.

### Encoder

Consists of multi-head attention and feed-forward network layers.

## Training

To train the Graphormer model, you need to prepare your dataset and define the training loop. Below is a simplified example:

```python
from zincdata import ZincDataset
from configuration import Config
from phormerModel import Graphormer

cfg = Config()
dataset = ZincDataset(cfg=cfg)
model = Graphormer(cfg)

# Define your optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
loss_fn = torch.nn.MSELoss()

# Training loop
for epoch in range(num_epochs):
    for batch in dataset.train_loader:
        # Prepare input data
        node_feat, in_degree, out_degree, path_data, dist, eigenvecs, eigen_value = batch

        # Forward pass
        output = model(node_feat, in_degree, out_degree, path_data, dist, eigenvecs, eigen_value)

        # Compute loss
        loss = loss_fn(output, target)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## Plotting

You can visualize the eigenvectors of a graph using the `plot_eigenvector.py` script. This script uses Plotly to create interactive plots.

```bash
python plot_eigenvector.py
```

## License

This project is licensed under the MIT License.
