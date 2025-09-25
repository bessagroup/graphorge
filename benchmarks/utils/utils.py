from pathlib import Path
import json
import numpy as np

from urllib.request import urlretrieve
from tqdm.notebook import tqdm
import logging
import pickle

from itertools import count


class DownloadProgressBar(tqdm):
    """
    Provides a progress bar compatible with the urlretrieve hook.

    Credit: https://github.com/tqdm/tqdm?tab=readme-ov-file#hooks-and-callbacks

    """

    def update_to(self, n_block, block_size, total_size=None):
        """
        Updates the progress bar with the number of blocks transferred.

        Parameters
        ----------
        n_block : int, optional
            Number of blocks transferred so far.
        block_size : int, optional
            Size of each block in bytes.
        total_size : int, optional
            Total size.

        """

        if total_size is not None:
            self.total = total_size

        # update() also sets self.n = n_block * block_size
        self.update(n_block * block_size - self.n)


def download_file(file, base_url, dest_path, overwrite=False):
    """
    Downloads a file from a given URL to a specified destination path.

    If the file already exists, download is skipped.

    Parameters
    ----------
    file : str
        The name of the file to download.

    base_url : str
        The base URL from which to download the file.

    dest_path : str
        The destination path where the file will be saved.

    """

    # Build the full URL
    url = f"{base_url}/{file}"

    # Validate the destination path
    dest_path = Path(dest_path).resolve()

    # Check if the file already exists and use it if overwrite is False
    if (dest_path / file).exists() and not overwrite:
        # Logging
        logging.info(
            msg=f"Using '{file}' cached in in {dest_path.relative_to(Path.cwd())}"
        )
        return

    # Create the destination directory if it doesn't exist
    dest_path.mkdir(parents=True, exist_ok=True)

    # Logging
    logging.info(msg=f"Downloading '{file}' to {dest_path}")

    # Download the file with progress bar
    with DownloadProgressBar(
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        miniters=1,
        desc=file,
    ) as pbar:
        urlretrieve(
            url,
            filename=dest_path / file,
            reporthook=pbar.update_to,
            data=None,
        )

        # Update the progress bar to reflect the total size in bytes since
        # total_size's units does not seem to be in bytes
        pbar.total = pbar.n


def get_critical_buckling_wavelength(poisson, shell_radius, shell_thickness):
    """Compute shell critical buckling wavelength.

    Parameters
    ----------
    poisson : float
        Material Poisson ratio.
    shell_radius : float
        Shell radius.
    shell_thickness : float
        Shell thickness.

    Returns
    -------
    critical_bw : float
        Critical buckling wavelength.
    """
    # Compute critical buckling wavelength
    critical_bw = (
        2
        * np.pi
        * ((12 * (1 - poisson**2)) ** (-1 / 4))
        * ((shell_radius * shell_thickness) ** (1 / 2))
    )
    return critical_bw


def parse_tensorflow_dataset(dataset_name, dataset_directory):
    """
    Parses a TensorFlow dataset and returns the features.

    Parameters
    ----------
    dataset : tf.data.Dataset
        The TensorFlow dataset to parse.

    """
    try:
        import tensorflow as tf
    except ImportError:
        err_msg = (
            "TensorFlow is required to parse the dataset."
            " See https://www.tensorflow.org/install"
            " for installation instructions."
        )
        raise ImportError(err_msg)

    # Validate the dataset path
    dataset_directory = Path(dataset_directory).resolve()

    # Load the metadata
    with open(dataset_directory / "meta.json", "r") as file:
        metadata = json.load(file)

    # Prepare the dataset path
    dataset_path = dataset_directory / f"{dataset_name}.tfrecord"

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset '{dataset_name}' not found in {dataset_directory}."
        )

    # Load the dataset
    dataset = tf.data.TFRecordDataset(dataset_path)

    # Parse the dataset output path
    output_path = dataset_directory / dataset_name
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize a sample counter
    sample_cnt = count()
    # Iterate over the dataset
    for sample in tqdm(dataset):
        # Parse the sample
        sample_data = _parse_tensorflow_sample(sample, metadata)
        # Prepare a sample path
        sample_path = output_path / f"{dataset_name}_{next(sample_cnt)}.pkl"
        # Export the sample to a pickle file
        with open(sample_path, "wb") as file:

            pickle.dump(sample_data, file)


def _parse_tensorflow_sample(sample, metadata):
    """
    Parses a single sample from the TensorFlow dataset.

    Credit: https://github.com/google-deepmind/deepmind-research/blob/master/meshgraphnets/dataset.py

    Parameters
    ----------
    sample : tf.train.Example
        The sample to parse.

    metadata : dict
        The metadata of the dataset.

    """

    try:
        import tensorflow as tf
    except ImportError:
        err_msg = (
            "TensorFlow is required to parse the dataset."
            " See https://www.tensorflow.org/install"
            " for installation instructions."
        )
        raise ImportError(err_msg)

    feature_dict = {k: tf.io.VarLenFeature(tf.string) for k in metadata["field_names"]}
    features = tf.io.parse_single_example(sample, feature_dict)
    sample_data = {}
    for key, field in metadata["features"].items():
        data = tf.io.decode_raw(features[key].values, getattr(tf, field["dtype"]))
        data = tf.reshape(data, field["shape"])
        if field["type"] == "static":
            data = data[0]

        elif field["type"] != "dynamic":
            raise ValueError("invalid data format")
        sample_data[key] = data.numpy()

    return sample_data


plotly_layout = dict(
    template="plotly_white",
    font_family="Arial",
    font_size=10,
    xaxis_showgrid=False,
    xaxis_ticks="inside",
    xaxis_mirror="allticks",
    xaxis_zeroline=False,
    xaxis_showline=True,
    xaxis_linecolor="black",
    yaxis_showgrid=False,
    yaxis_ticks="inside",
    yaxis_mirror="allticks",
    yaxis_zeroline=False,
    yaxis_showline=True,
    yaxis_linecolor="black",
    colorway=["#4477aa", "#ef6f7f", "#288b39", "#288b39", "#aa3377", "#000000"],
    xaxis_title_standoff=3,
    yaxis_title_standoff=3,
    margin=dict(l=40, r=10, t=10, b=40),
    yaxis_automargin=True,
)
"""A Plotly layout dictionary for consistent figure styling."""


def plot_interactive_graph(graph_data):
    """
    Plots the graph using Plotly.

    Parameters
    ----------
    graph_data : GraphData
        The graph data to plot.
    """
    import plotly.graph_objects as go

    # Create a 3D scatter plot for nodes and lines for edges
    figure = go.Figure()

    # Extract node coordinates
    node_coords = graph_data.get_nodes_coords()

    if node_coords.shape[1] == 3:
        # 3D graph
        is_3d = True
    else:
        is_3d = False

    # Prepare edge coordinates by indexing node coordinates with edge indexes
    edge_x = node_coords[graph_data.get_graph_edges_indexes()][:, :, 0].ravel()

    # We need to insert None values between edges to avoid connecting them
    # in the plotly scatter3d while adding only one trace
    insert_index = np.arange(0, edge_x.shape[0], 2)
    edge_x = np.insert(edge_x, insert_index, None)

    # Prepare y edge coordinates
    edge_y = node_coords[graph_data.get_graph_edges_indexes()][:, :, 1].ravel()
    edge_y = np.insert(edge_y, insert_index, None)

    # Prepare z edge coordinates
    if is_3d:
        edge_z = node_coords[graph_data.get_graph_edges_indexes()][:, :, 2].ravel()
        edge_z = np.insert(edge_z, insert_index, None)

    # Prepare hovertemplate for edge features, if any
    edge_feature_names = graph_data.get_metadata().get("edge_feature_names", [])
    hovertemplate_edge_features = "".join(
        [
            f"<br>{name}: %{{customdata[{i}]:.2f}}"
            for i, name in enumerate(edge_feature_names)
        ]
    )

    # Add edges
    if is_3d:
        figure.add_scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            mode="lines",
            line_color="#ef6f7f",
            line_width=2,
            name="Edges",
            hoverinfo="none",
        )

        if hovertemplate_edge_features:
            # Add invisible scatter for edge features hover at mid edge position
            mid_edge_x = node_coords[graph_data.get_graph_edges_indexes()][
                :, :, 0
            ].mean(axis=1)
            mid_edge_y = node_coords[graph_data.get_graph_edges_indexes()][
                :, :, 1
            ].mean(axis=1)
            mid_edge_z = node_coords[graph_data.get_graph_edges_indexes()][
                :, :, 2
            ].mean(axis=1)

            figure.add_scatter3d(
                x=mid_edge_x,
                y=mid_edge_y,
                z=mid_edge_z,
                mode="markers",
                marker_size=0.000000001,
                marker_color="#ef6f7f",
                line_color="#ef6f7f",
                hovertemplate=f"{hovertemplate_edge_features}<extra></extra>",
                customdata=graph_data.get_edge_features_matrix(),
                showlegend=False,
            )

    else:
        figure.add_scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line_color="#ef6f7f",
            line_width=2,
            name="Edges",
            hoverinfo="none",
        )

        if hovertemplate_edge_features:
            # Add invisible scatter for edge features hover at mid edge position
            mid_edge_x = node_coords[graph_data.get_graph_edges_indexes()][
                :, :, 0
            ].mean(axis=1)
            mid_edge_y = node_coords[graph_data.get_graph_edges_indexes()][
                :, :, 1
            ].mean(axis=1)

            figure.add_scatter(
                x=mid_edge_x,
                y=mid_edge_y,
                mode="markers",
                marker_size=0.000000001,
                marker_color="#ef6f7f",
                line_color="#ef6f7f",
                hovertemplate=f"{hovertemplate_edge_features}<extra></extra>",
                customdata=graph_data.get_edge_features_matrix(),
                showlegend=False,
            )

    # Prepare hovertemplate for node features, if any
    node_feature_names = graph_data.get_metadata().get("node_feature_names", [])
    hovertemplate_features = "".join(
        [
            f"<br>{name}: %{{customdata[{i}]:.2f}}"
            for i, name in enumerate(node_feature_names)
        ]
    )

    # Add nodes
    if is_3d:
        figure.add_scatter3d(
            x=node_coords[:, 0],
            y=node_coords[:, 1],
            z=node_coords[:, 2],
            text=[f"Node {i}" for i in range(graph_data.get_n_node())],
            mode="markers",
            marker_color="#4477aa",
            marker_line_color="black",
            marker_line_width=1,
            marker_size=10,
            customdata=graph_data.get_node_features_matrix(),
            hovertemplate=(
                "%{text}<br>x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}"
                + hovertemplate_features
                + "<extra></extra>"
            ),
            name="Nodes",
        )

        figure.update_layout(
            **plotly_layout, scene_aspectmode="data", width=800, height=400
        )

    else:
        figure.add_scatter(
            x=node_coords[:, 0],
            y=node_coords[:, 1],
            text=[f"Node {i}" for i in range(graph_data.get_n_node())],
            mode="markers",
            marker_color="#4477aa",
            marker_line_color="black",
            marker_line_width=1,
            marker_size=10,
            customdata=graph_data.get_node_features_matrix(),
            hovertemplate=(
                "%{text}<br>x: %{x:.2f}<br>y: %{y:.2f}"
                + hovertemplate_features
                + "<extra></extra>"
            ),
            name="Nodes",
        )

        figure.update_layout(**plotly_layout, width=500, height=500)

        figure.update_yaxes(
            scaleanchor="x",
            scaleratio=1,
        )

    return figure


def graph_to_pyvista_mesh(graph):
    import pyvista as pv

    # Get the mesh vertices, add a third dimension with zeros to make it 3D
    mesh_vertices = np.insert(arr=graph.pos.numpy(), obj=2, values=0, axis=1)

    # Get the raw sample data which define the mesh cells
    raw_data_file = (
        Path("data")
        / graph.metadata["dataset_name"]
        / f"{graph.metadata['dataset_name']}"
        f"_sample_{graph.metadata['sample_id']}.pkl"
    )
    with open(raw_data_file, "rb") as file:
        sample = pickle.load(file)

    # Extract the cells
    cells = sample["cells"]

    # The cell are already defined as a list of 3 vertices
    # Add the number of vertices to the beginning of each cell to comply with the
    # vtk format
    vtk_faces = np.insert(arr=cells, obj=0, values=3, axis=1)

    # Create the mesh
    mesh = pv.PolyData(mesh_vertices, faces=vtk_faces)

    # Add the velocity as point data
    mesh.point_data["velocity"] = graph.x[:, :2].numpy()

    # Add the pressure as point data
    mesh.point_data["pressure"] = graph.x[:, 2:3].numpy()

    return mesh
