"""Setting fixtures for pytest."""
#
#                                                                       Modules
# =============================================================================
# Third-party
import pytest
import numpy as np
import torch
# Local
from src.graphorge.gnn_base_model.data.graph_data import GraphData
from src.graphorge.gnn_base_model.model.gnn_model import GNNEPDBaseModel
# =============================================================================
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
# =============================================================================
#
# =============================================================================
@pytest.fixture
def graph_patch_data_2d():
    """GNN-based material patch graph data for 2D material patch."""
    # Set sample ID
    sample_id = 0
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set number of spatial dimensions
    n_dim = 2
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set number of nodes
    n_nodes = 10
    # Set number of edges
    n_edges = 5
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set coordinates of nodes
    nodes_coords = np.random.rand(n_nodes, n_dim)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set number of node input and output features
    n_node_in = 2
    n_node_out = 5
    # Set number of edge input and output features
    n_edge_in = 3
    n_edge_out = 4
    # Set number of global input and output features
    n_global_in = 3
    n_global_out = 2
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Instantiate GNN-based material patch graph data
    gnn_patch_data = GraphData(n_dim=n_dim, nodes_coords=nodes_coords)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate random edges indexes
    edges_indexes = np.random.choice(n_nodes, size=(n_edges, 2), replace=False)    
    # Set material patch graph edges indexes
    gnn_patch_data.set_graph_edges_indexes(edges_indexes_mesh=edges_indexes)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate and set random nodes features input matrix
    node_features_in = np.random.rand(n_nodes, n_node_in)
    gnn_patch_data.set_node_features_matrix(node_features_in)
    # Generate and set random edges features input matrix
    edge_features_in = np.random.rand(n_edges, n_edge_in)
    gnn_patch_data.set_edge_features_matrix(edge_features_in)
    # Generate and set random global features input matrix
    global_features_in = np.random.rand(1, n_global_in)
    gnn_patch_data.set_global_features_matrix(global_features_in)
    # Generate and set random nodes targets matrix
    node_targets_matrix = np.random.rand(n_nodes, n_node_out)
    gnn_patch_data.set_node_targets_matrix(node_targets_matrix)
    # Generate and set random edges targets matrix
    edge_targets_matrix = np.random.rand(n_edges, n_edge_out)
    gnn_patch_data.set_edge_targets_matrix(edge_targets_matrix)
    # Generate and set random global targets matrix
    global_targets_matrix = np.random.rand(1, n_global_out)
    gnn_patch_data.set_global_targets_matrix(global_targets_matrix)
    # Set sample ID metadata
    metadata = dict(sample_id=sample_id)
    gnn_patch_data.set_metadata(metadata)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return gnn_patch_data
# -----------------------------------------------------------------------------
@pytest.fixture
def batch_graph_patch_data_2d():
    """Batch of GNN-based material patch graph data for 2D material patches."""
    # Initialize list of GNN-based material patch graph data
    batch_graph_data = []
    # Set number of samples
    n_sample = 5
    # Set number of spatial dimensions
    n_dim = 2
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over samples
    for i in range(n_sample):
        # Set random number of nodes
        n_nodes = int(torch.randint(low=10, high=20, size=(1,)))
        # Set number of edges
        n_edges = int(torch.randint(low=2, high=5, size=(1,)))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set coordinates of nodes
        nodes_coords = np.random.rand(n_nodes, n_dim)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set number of node input and output features
        n_node_in = 2
        n_node_out = 5
        # Set number of edge input and output features
        n_edge_in = 3
        n_edge_out = 4
        # Set number of global input and output features
        n_global_in = 3
        n_global_out = 2
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Instantiate GNN-based material patch graph data
        gnn_patch_data = GraphData(n_dim=n_dim, nodes_coords=nodes_coords)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Generate random edges indexes
        edges_indexes = np.random.choice(n_nodes, size=(n_edges, 2),
                                         replace=False)    
        # Set material patch graph edges indexes
        gnn_patch_data.set_graph_edges_indexes(
            edges_indexes_mesh=edges_indexes)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Generate and set random nodes features input matrix
        node_features_in = np.random.rand(n_nodes, n_node_in)
        gnn_patch_data.set_node_features_matrix(node_features_in)
        # Generate and set random edges features input matrix
        edge_features_in = np.random.rand(n_edges, n_edge_in)
        gnn_patch_data.set_edge_features_matrix(edge_features_in)
        # Generate and set random global features input matrix
        global_features_in = np.random.rand(1, n_global_in)
        gnn_patch_data.set_global_features_matrix(global_features_in)
        # Generate and set random nodes targets matrix
        node_targets_matrix = np.random.rand(n_nodes, n_node_out)
        gnn_patch_data.set_node_targets_matrix(node_targets_matrix)
        # Generate and set random edges targets matrix
        edge_targets_matrix = np.random.rand(n_edges, n_edge_out)
        gnn_patch_data.set_edge_targets_matrix(edge_targets_matrix)
        # Generate and set random global targets matrix
        global_targets_matrix = np.random.rand(1, n_global_out)
        gnn_patch_data.set_global_targets_matrix(global_targets_matrix)
        # Set sample ID metadata
        metadata = dict(sample_id=i)
        gnn_patch_data.set_metadata(metadata)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Store GNN-based material patch graph data
        batch_graph_data.append(gnn_patch_data)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return batch_graph_data
# -----------------------------------------------------------------------------
@pytest.fixture
def gnn_epd_base_model(tmp_path):
    """GNN Encoder-Processor-Decoder base model."""
    # Set GNN Encoder-Processor-Decoder base model initialization parameters
    model_init_args = dict(n_node_in=2, n_node_out=5, n_edge_in=3,
                           n_edge_out=4, n_global_in=3, n_global_out=2,
                           n_message_steps=2, enc_n_hidden_layers=2,
                           pro_n_hidden_layers=3, dec_n_hidden_layers=4,
                           hidden_layer_size=2, model_directory=tmp_path,
                           model_name='graph_neural_network_model',
                           is_model_in_normalized=False,
                           is_model_out_normalized=False,
                           enc_node_hidden_activ_type='relu',
                           enc_node_output_activ_type='identity',
                           enc_edge_hidden_activ_type='relu',
                           enc_edge_output_activ_type='identity',
                           enc_global_hidden_activ_type='relu',
                           enc_global_output_activ_type='identity',
                           pro_node_hidden_activ_type='relu',
                           pro_node_output_activ_type='identity',
                           pro_edge_hidden_activ_type='relu',
                           pro_edge_output_activ_type='identity',
                           pro_global_hidden_activ_type='relu',
                           pro_global_output_activ_type='identity',
                           dec_node_hidden_activ_type='relu',
                           dec_node_output_activ_type='identity',
                           dec_edge_hidden_activ_type='relu',
                           dec_edge_output_activ_type='identity',
                           dec_global_hidden_activ_type='relu',
                           dec_global_output_activ_type='identity',
                           )
    # Build GNN Encoder-Processor-Decoder base model
    model = GNNEPDBaseModel(**model_init_args)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return model
# -----------------------------------------------------------------------------
@pytest.fixture
def gnn_epd_base_model_norm(tmp_path, batch_graph_patch_data_2d):
    """GNN Encoder-Processor-Decoder base model with data normalization."""
    # Pick material patch graph data
    graph_data = batch_graph_patch_data_2d[0]
    # Get material patch graph input features matrices
    node_features_in = graph_data.get_node_features_matrix()
    edge_features_in = graph_data.get_edge_features_matrix()
    global_features_in = graph_data.get_global_features_matrix()
    # Get number of node input and output features
    n_node_in = node_features_in.shape[1]
    n_node_out = graph_data.get_node_targets_matrix().shape[1]
    # Get number of edge input and output features
    n_edge_in = edge_features_in.shape[1]
    n_edge_out = graph_data.get_edge_targets_matrix().shape[1]
    # Get number of global input and output features
    n_global_in = global_features_in.shape[1]
    n_global_out = graph_data.get_global_targets_matrix().shape[1]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build GNN Encoder-Processor-Decoder base model
    model_init_args = dict(n_node_in=n_node_in, n_node_out=n_node_out,
                           n_edge_in=n_edge_in, n_edge_out=n_edge_out,
                           n_global_in=n_global_in, n_global_out=n_global_out,
                           n_message_steps=2,
                           enc_n_hidden_layers=2, pro_n_hidden_layers=3,
                           dec_n_hidden_layers=4, hidden_layer_size=2,
                           model_directory=str(tmp_path),
                           is_model_in_normalized=True,
                           is_model_out_normalized=True,
                           enc_node_hidden_activ_type='relu',
                           enc_node_output_activ_type='identity',
                           enc_edge_hidden_activ_type='relu',
                           enc_edge_output_activ_type='identity',
                           enc_global_hidden_activ_type='relu',
                           enc_global_output_activ_type='identity',
                           pro_node_hidden_activ_type='relu',
                           pro_node_output_activ_type='identity',
                           pro_edge_hidden_activ_type='relu',
                           pro_edge_output_activ_type='identity',
                           pro_global_hidden_activ_type='relu',
                           pro_global_output_activ_type='identity',
                           dec_node_hidden_activ_type='relu',
                           dec_node_output_activ_type='identity',
                           dec_edge_hidden_activ_type='relu',
                           dec_edge_output_activ_type='identity',
                           dec_global_hidden_activ_type='relu',
                           dec_global_output_activ_type='identity',
                           )
    model = GNNEPDBaseModel(**model_init_args)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build dataset
    dataset = [gnn_patch_data.get_torch_data_object()
               for gnn_patch_data in batch_graph_patch_data_2d]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Fit model data scalers
    model.fit_data_scalers(dataset)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return model
# -----------------------------------------------------------------------------
@pytest.fixture
def pytorch_optimizer_adam(gnn_epd_base_model):
    """PyTorch Adam optimizer for GNN EPD base model parameters."""
    # Set GNN Encoder-Processor-Decoder base model
    model = gnn_epd_base_model
    # Set Adam optimizer for model parameters
    optimizer = torch.optim.Adam(model.parameters(recurse=True))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return optimizer