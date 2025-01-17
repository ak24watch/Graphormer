import igraph as ig
import plotly.graph_objects as go
from zincdata import ZincDataset
from configuration import Config


def plot_eigenvector(graph, eigenvector):
    """
    Plot the eigenvector for a given graph, including both nodes and edges.

    Args:
        graph: The graph object.
        eigenvector: The eigenvector to be plotted.
    """
    # Generate 2D coordinates for nodes using Kamada-Kawai layout
    layout = graph.layout("kk")
    x_coords, y_coords = zip(*layout.coords)

    # Prepare edge coordinates
    edge_x = []
    edge_y = []
    for edge in graph.es:
        # Get the positions of the source and target nodes
        x0, y0 = layout[edge.source]
        x1, y1 = layout[edge.target]

        # Append to the edge coordinate lists
        edge_x.extend([x0, x1, None])  # None creates the break between edge points
        edge_y.extend([y0, y1, None])

    # Edge trace (for the graph's edges)
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
        mode="lines",
    )

    # Node trace (for the graph's nodes)
    node_trace = go.Scatter(
        x=x_coords,
        y=y_coords,
        mode="markers",
        marker=dict(
            size=10,
            color=eigenvector,
            colorscale="Viridis",
            colorbar=dict(title="Eigenvector Value"),
            line_width=2,
        ),
    )

    # Create the Plotly figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title="Graph Eigenvector Plot",
            showlegend=False,
            hovermode="closest",
            margin=dict(b=0, l=0, r=0, t=40),
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
        ),
    )

    # Show the plot interactively
    fig.show()

    # Optionally, save the plot as a static image
    fig.write_image("eigenvector_plot.png")


if __name__ == "__main__":
    cfg = Config()
    dataset = ZincDataset(cfg=cfg)
    graph, _ = dataset.train_samples[0]  # Assuming the dataset provides graph data

    # Convert to igraph object
    ig_graph = ig.Graph(directed=False)
    ig_graph.add_vertices(graph.number_of_nodes())
    ig_graph.add_edges([(int(edge[0]), int(edge[1])) for edge in graph.edges()])

    # Extract eigenvector values from the graph
    eigenvector = graph.ndata["eigenvec"][
        :, 0
    ].tolist()  # Assuming eigenvector is stored here
    print
    # Plot the graph with eigenvector values
    plot_eigenvector(ig_graph, eigenvector)
