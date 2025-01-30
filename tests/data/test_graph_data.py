"""Test Graph Data components."""
#
#                                                                       Modules
# =============================================================================
# Third-party
import numpy as np
import pytest
# Local
from src.graphorge.gnn_base_model.data.graph_data import GraphData
# =============================================================================
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Guillaume Broggi (g.broggi@tudelft.nl)'
__credits__ = ['Guillaume Broggi', ]
__status__ = 'Planning'
# =============================================================================
#
# =============================================================================

@pytest.fixture
def graph_data(n_dim=2, n_nodes=10):
    """Return a graph data object."""
    # Sample random nodes coordinates
    nodes_coords = np.random.rand(n_nodes, n_dim)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Instantiate a graph data object
    return GraphData(n_dim=n_dim, nodes_coords=nodes_coords)
# -----------------------------------------------------------------------------
def test_graph_data_set_graph_edges_indexes_unique(graph_data):
    """Test setting unique graph edges indexes."""
    # Generate random edge indices
    edge_indices = np.random.randint(0, graph_data.get_n_node(), (10, 2))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set the edge indices
    graph_data.set_graph_edges_indexes(edges_indexes_mesh=edge_indices)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Assert edge indices are unique
    _, unique_counts = np.unique(graph_data.get_graph_edges_indexes(), axis=0,
                                 return_counts=True)
    assert (unique_counts == 1).all()
# -----------------------------------------------------------------------------
def test_graph_data_set_graph_edges_indexes_non_unique(graph_data):
    """Test setting non-unique graph edges indexes."""
    # Generate random edge indices
    edge_indices = np.random.randint(0, graph_data.get_n_node(), (10, 2))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set the edge indices
    graph_data.set_graph_edges_indexes(edges_indexes_mesh=edge_indices,
                                       is_unique=False)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Assert the input remained unchanged
    assert (graph_data.get_graph_edges_indexes() == edge_indices).all()