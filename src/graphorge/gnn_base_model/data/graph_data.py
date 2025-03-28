"""Graph Neural Network graph data.

Classes
-------
GraphData
    Graph Neural Network graph data.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import os
import copy
# Third-party
import numpy as np
import scipy.spatial
import matplotlib.pyplot as plt
import torch
import torch_geometric.data
# Local
from ioput.iostandard import new_file_path_with_int
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
# =============================================================================
#
# =============================================================================
class GraphData:
    """Graph Neural Network graph data.
    
    Attributes
    ----------
    _n_dim : int
        Number of spatial dimensions.
    _nodes_coords : numpy.ndarray(2d)
        Coordinates of nodes stored as a numpy.ndarray(2d) with shape
        (n_nodes, n_dim). Coordinates of i-th node are stored in
        nodes_coords[i, :].
    _n_node : int
        Number of nodes.
    _n_edge : int
        Number of edges.
    _node_features_matrix : numpy.ndarray(2d)
        Nodes input features matrix stored as a numpy.ndarray(2d) of shape
        (n_nodes, n_features).
    _edge_features_matrix : numpy.ndarray(2d)
        Edges input features matrix stored as a numpy.ndarray(2d) of shape
        (n_edges, n_features).
    _global_features_matrix : numpy.ndarray(2d)
        Global input features matrix stored as a numpy.ndarray(2d) of shape
        (1, n_features).
    _node_targets_matrix : numpy.ndarray(2d)
        Nodes targets matrix stored as a numpy.ndarray(2d) of shape
        (n_nodes, n_targets).
    _edge_targets_matrix : numpy.ndarray(2d)
        Edges targets matrix stored as a numpy.ndarray(2d) of shape
        (n_nodes, n_targets).
    _global_targets_matrix : numpy.ndarray(2d)
        Global targets matrix stored as a numpy.ndarray(2d) of shape
        (1, n_targets).
    _edges_indexes : numpy.ndarray(2d)
        Edges indexes matrix stored as numpy.ndarray[int](2d) with shape
        (n_edges, 2), where the i-th edge is stored in
        edges_indexes[i, :] as (start_node_index, end_node_index).
    _metadata : dict
        A dictionary containing any metadata information.
    
    Methods
    -------
    get_torch_data_object(self)
        Get PyG homogeneous graph data object.
    get_nodes_coords(self)
        Get graph node coordinates.
    get_n_node(self)
        Get graph number of nodes.
    get_n_edge(self)
        Get graph number of edges.
    set_graph_edges_indexes(self, connect_radius=None, \
                            edges_indexes_mesh=None, is_unique=True)
        Set graph edges indexes.
    get_graph_edges_indexes(self)
        Get graph edges indexes.
    set_node_features_matrix(self, node_features_matrix)
        Set nodes input features matrix.
    get_node_features_matrix(self)
        Set nodes input features matrix.
    set_edge_features_matrix(self, edge_features_matrix)
        Set edges input features matrix.
    get_edge_features_matrix(self)
        Get edges input features matrix.
    set_global_features_matrix(self, global_features_matrix)
        Set global input features matrix.
    get_global_features_matrix(self)
        Get global input features matrix.  
    set_node_targets_matrix(self, node_targets_matrix)
        Set node targets matrix.
    get_node_targets_matrix(self)
        Get node targets matrix.  
    set_edge_targets_matrix(self, edge_targets_matrix)
        Set edge targets matrix.   
    get_edge_targets_matrix(self)
        Get edge targets matrix.
    set_global_targets_matrix(self, global_targets_matrix)
        Set global targets matrix.
    get_global_targets_matrix(self)
        Get global targets matrix.
    set_metadata(self, metadata)
        Set metadata information.
    add_metadata(self, key, value)
        Add metadata information.
    get_metadata(self)
        Get metadata information.
    plot_graph(self, is_show_plot=False, is_save_plot=False, \
               save_directory=None, plot_name=None, is_overwrite_file=False)
        Generate plot of graph.
    _get_edges_from_local_radius(nodes_coords, connect_radius)
        Get edges between nodes that are within a given connectivity radius.
    get_undirected_unique_edges(edges_indexes)
        Get set of undirected unique edges indexes.
    _check_edges_indexes_matrix(edges_indexes)
        Check if given edges indexes matrix is valid.
    """
    def __init__(self, n_dim, nodes_coords):
        """Constructor.
        
        Parameters
        ----------
        n_dim : int
            Number of spatial dimensions.
        nodes_coords : numpy.ndarray(2d)
            Coordinates of nodes stored as a numpy.ndarray(2d) with shape
            (n_nodes, n_dim). Coordinates of i-th node are stored in
            nodes_coords[i, :].
        """
        self._n_dim = n_dim
        self._nodes_coords = nodes_coords[:, :n_dim]
        self._n_node = self._nodes_coords.shape[0]
        # Initialize graph edges
        self._n_edge = None
        self._edges_indexes = None
        # Initialize features matrices
        self._node_features_matrix = None
        self._edge_features_matrix = None
        self._global_features_matrix = None
        self._node_targets_matrix = None
        self._edge_targets_matrix = None
        self._global_targets_matrix = None
        # Initialize metadata
        self._metadata = {}
    # -------------------------------------------------------------------------  
    def get_torch_data_object(self):
        """Get PyG homogeneous graph data object.
        
        Returns
        -------
        pyg_graph : torch_geometric.data.Data
            PyG data object describing a homogeneous graph.
        """
        # Set PyG number of nodes
        num_nodes = self._n_node
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set PyG node feature matrix
        x = None
        if self._node_features_matrix is not None:
            x = torch.tensor(copy.deepcopy(self._node_features_matrix),
                             dtype=torch.float)
        # Set PyG graph connectivity
        edge_index = None
        if self._edges_indexes is not None:
            edge_index = torch.tensor(
                np.transpose(copy.deepcopy(self._edges_indexes)),
                dtype=torch.long)
        # Set PyG edge feature matrix
        edge_attr = None
        if self._edge_features_matrix is not None:
            edge_attr = torch.tensor(copy.deepcopy(self._edge_features_matrix),
                                     dtype=torch.float)
        # Set PyG node ground-truth labels
        y = None
        if self._node_targets_matrix is not None:
            y = torch.tensor(copy.deepcopy(self._node_targets_matrix),
                             dtype=torch.float)
        # Set PyG node position matrix
        pos = None
        if self._nodes_coords is not None:
            pos = torch.tensor(copy.deepcopy(self._nodes_coords),
                               dtype=torch.float)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set PyG custom attributes:
        # Set edge ground-truth labels
        edge_targets_matrix = None
        if self._edge_targets_matrix is not None:
            edge_targets_matrix = \
                torch.tensor(copy.deepcopy(self._edge_targets_matrix),
                             dtype=torch.float)
        # Set global attributes
        global_features_matrix = None
        if self._global_features_matrix is not None:
            global_features_matrix = \
                torch.tensor(copy.deepcopy(self._global_features_matrix),
                             dtype=torch.float)
        # Set global ground-truth labels
        global_targets_matrix = None
        if self._global_targets_matrix is not None:
            global_targets_matrix = \
                torch.tensor(copy.deepcopy(self._global_targets_matrix),
                             dtype=torch.float)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set metadata
        metadata = {}
        if isinstance(self._metadata, dict):
            # Iterate over metadata items
            for key, value in self._metadata.items():
                    # Cast to a tensor if possible
                    try:
                        metadata[key] = torch.tensor(copy.deepcopy(value))
                    # If not, copy the value (e.g., strings)
                    except:
                        metadata[key] = copy.deepcopy(value)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Instantiate PyG homogeneous graph data object
        pyg_graph = torch_geometric.data.Data(
            x=x, edge_index=edge_index, edge_attr=edge_attr, y=y,
            pos=pos, num_nodes=num_nodes,
            edge_targets_matrix=edge_targets_matrix,
            global_features_matrix=global_features_matrix,
            global_targets_matrix=global_targets_matrix,
            metadata=metadata)
        # Validate graph data object
        pyg_graph.validate(raise_on_error=True)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return pyg_graph
    # -------------------------------------------------------------------------
    @staticmethod
    def extract_data_torch_data_object(pyg_graph, attributes):
        """Extract data from PyG homogeneous graph data object.
        
        Parameters
        ----------
        pyg_graph : torch_geometric.data.Data
            PyG data object describing a homogeneous graph.
        attributes : tuple
            Attributes to be extracted from PyG data object. Available
            attributes coincide with the attributes of GraphData.

        Returns
        -------
        attributes_data : tuple
            Extracted attributes data sorted according with provided attributes
            sequence.
        """
        # Check input graph
        if not isinstance(pyg_graph, torch_geometric.data.Data):
            raise RuntimeError('Input graph is not torch_geometric.data.Data.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set available attributes
        available_attributes = \
            ('n_dim', 'n_node', 'n_edge', 'nodes_coords', 'edge_indexes',
             'node_features_matrix', 'edge_features_matrix',
             'global_features_matrix', 'node_targets_matrix',
             'edge_targets_matrix', 'global_targets_matrix', 'metadata')
        
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize extracted attributes data
        attributes_data = []
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over attributes
        for attribute in attributes:
            # Extract attribute
            if attribute == 'n_dim':
                data = int(pyg_graph.pos.shape[1])
            elif attribute == 'n_node':
                data = int(pyg_graph.num_nodes)
            elif attribute == 'n_edge':
                data = int(pyg_graph.edge_index.shape[0])
            elif attribute == 'nodes_coords':
                data = pyg_graph.pos.numpy()
            elif attribute == 'edge_indexes':
                data = pyg_graph.edge_index.numpy().transpose()
            elif attribute == 'node_features_matrix':
                data = pyg_graph.x.numpy()
            elif attribute == 'edge_features_matrix':
                data = pyg_graph.edge_attr.numpy()
            elif attribute == 'global_features_matrix':
                data = pyg_graph.global_features_matrix.numpy()
            elif attribute == 'node_targets_matrix':
                data = pyg_graph.y.numpy()
            elif attribute == 'edge_targets_matrix':
                data = pyg_graph.edge_targets_matrix.numpy()
            elif attribute == 'global_targets_matrix':
                data = pyg_graph.global_targets_matrix.numpy()
            elif attribute == 'metadata':
                data = pyg_graph.metadata
            else:
                raise RuntimeError(f'The attribute {attribute} is not '
                                   f'available. Available attributes:\n\n'
                                   f'{available_attributes}')
            # Store extracted attribute
            attributes_data.append(data)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return tuple(attributes_data)
    # -------------------------------------------------------------------------
    def set_graph_edges_indexes(self, connect_radius=None,
                                edges_indexes_mesh=None, is_unique=True):
        """Set graph edges indexes and number of edges.
        
        Sets existing edges input features and targets matrices to None.
        
        Parameters
        ----------
        connect_radius : float, default=None
            Connectivity radius that sets the maximum distance between two
            nodes that leads to an edge. If None, then no edges are generated
            from distance-based search.
        edges_indexes_mesh : numpy.ndarray(2d), default=None
            Edges stemming from any relevant mesh representation (e.g., finite
            element mesh) and that should be accounted for. Edges indexes
            matrix stored as numpy.ndarray[int](2d) with shape (n_edges, 2),
            where the i-th edge is stored in edges_indexes[i, :] as
            (start_node_index, end_node_index).
        is_unique : bool, default=True
            Remove any existent duplicated edges if `True`. Resulting unique 
            edges are sorted by ascending order of the corresponding indexes,
            according to 'numpy.unique' output. If `False`, edges are kept
            unmodified.
        """
        # Initialize edges indexes
        edges_indexes = np.empty((0, 2), dtype=int)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get edges from distance-based search within given connectivity radius
        if connect_radius is not None:
            edge_indexes_radius = type(self)._get_edges_from_local_radius(
                nodes_coords=self._nodes_coords, connect_radius=connect_radius)
            # Append distance-based edges indexes
            edges_indexes = \
                np.concatenate((edges_indexes, edge_indexes_radius), axis=0)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Append given edges stemming from mesh representation
        if edges_indexes_mesh is not None:
            # Check mesh-based edges indexes
            GraphData._check_edges_indexes_matrix(edges_indexes_mesh)
            # Append mesh-based edges indexes
            edges_indexes = np.concatenate((edges_indexes, edges_indexes_mesh),
                                           axis=0)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Remove any existent duplicated edges
        if is_unique:
            edges_indexes =np.unique(edges_indexes, axis=0)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set number of edges
        self._n_edge = edges_indexes.shape[0]
        # Set edges indexes
        self._edges_indexes = edges_indexes
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize edges input features and targets matrices
        self._edge_features_matrix = None
        self._edge_targets_matrix = None
    # -------------------------------------------------------------------------
    def get_nodes_coords(self):
        """Get graph node coordinates.
        
        Returns
        -------
        nodes_coords : numpy.ndarray(2d), default=None
            Coordinates of nodes stored as a numpy.ndarray(2d) with shape
            (n_nodes, n_dim). Coordinates of i-th node are stored in
            nodes_coords[i, :].
        """
        return copy.deepcopy(self._nodes_coords)
    # -------------------------------------------------------------------------
    def get_n_node(self):
        """Get graph number of nodes.
        
        Returns
        -------
        n_node : int
            Number of nodes.
        """
        return self._n_node
    # -------------------------------------------------------------------------
    def get_n_edge(self):
        """Get graph number of edges.
        
        Returns
        -------
        n_edge : int
            Number of edges.
        """
        return self._n_edge
    # -------------------------------------------------------------------------
    def get_graph_edges_indexes(self):
        """Get graph edges indexes.
        
        Returns
        -------
        edges_indexes : numpy.ndarray(2d)
            Edges indexes matrix stored as numpy.ndarray[int](2d) with shape
            (num_edges, 2), where the i-th edge is stored in
            edges_indexes[i, :] as (start_node_index, end_node_index).
        """
        return copy.deepcopy(self._edges_indexes)
    # -------------------------------------------------------------------------
    def set_node_features_matrix(self, node_features_matrix):
        """Set nodes input features matrix.
        
        Parameters
        ----------
        node_features_matrix : {numpy.ndarray(2d), None}
            Nodes input features matrix stored as a numpy.ndarray(2d) of shape
            (n_nodes, n_features).
        """
        if node_features_matrix is None:
            self._node_features_matrix = None
        else:
            # Check nodes input features matrix
            if not isinstance(node_features_matrix, np.ndarray):
                raise RuntimeError('Nodes input features matrix must be '
                                   'provided as a numpy 2d array of shape '
                                   '(n_nodes, n_features).')
            elif len(node_features_matrix.shape) != 2:
                raise RuntimeError('Nodes input features matrix must be '
                                   'provided as a numpy 2d array of shape '
                                   '(n_nodes, n_features).')
            elif node_features_matrix.shape[0] != self._n_node:
                raise RuntimeError(f'Nodes input features matrix shape '
                                   f'({self._nodes_coords.shape}) is not '
                                   f'compatible with number of nodes '
                                   f'of graph.'
                                   f'({self._n_node}).')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set nodes input features matrix
            self._node_features_matrix = copy.deepcopy(node_features_matrix)
    # -------------------------------------------------------------------------
    def get_node_features_matrix(self):
        """Set nodes input features matrix.
        
        Returns
        -------
        node_features_matrix : numpy.ndarray(2d)
            Nodes input features matrix stored as a numpy.ndarray(2d) of shape
            (n_nodes, n_features).
        """
        return copy.deepcopy(self._node_features_matrix)
    # -------------------------------------------------------------------------
    def set_edge_features_matrix(self, edge_features_matrix):
        """Set edges input features matrix.
        
        Parameters
        ----------
        edge_features_matrix : numpy.ndarray(2d)
            Edges input features matrix stored as a numpy.ndarray(2d) of shape
            (n_edges, n_features).
        """
        if edge_features_matrix is None:
            self._edge_features_matrix = None
        else:
            # Check edges input features matrix
            if not isinstance(edge_features_matrix, np.ndarray):
                raise RuntimeError('Edges input features matrix must be '
                                   'provided as a numpy 2d array of shape '
                                   '(n_edges, n_features).')
            elif len(edge_features_matrix.shape) != 2:
                raise RuntimeError('Edges input features matrix must be '
                                   'provided as a numpy 2d array of shape '
                                   '(n_edges, n_features).')
            elif (self._n_edge is not None
                  and (edge_features_matrix.shape[0] != self._n_edge)):
                raise RuntimeError(f'Edges input features matrix shape '
                                   f'({edge_features_matrix.shape}) is not '
                                   f'compatible with number of edges '
                                   f'of graph ({self._n_edge}).')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set edges input features matrix
            self._edge_features_matrix = copy.deepcopy(edge_features_matrix)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set unknown number of edges according with edges input features
            # matrix
            if self._n_edge is None:
                self._n_edge = self._edge_features_matrix.shape[0]
    # -------------------------------------------------------------------------
    def get_edge_features_matrix(self):
        """Get edges input features matrix.
        
        Returns
        -------
        edge_features_matrix : numpy.ndarray(2d)
            Edges input features matrix stored as a numpy.ndarray(2d) of shape
            (n_edges, n_features).
        """
        return copy.deepcopy(self._edge_features_matrix)
    # -------------------------------------------------------------------------   
    def set_global_features_matrix(self, global_features_matrix):
        """Set global input features matrix.
        
        Parameters
        ----------
        global_features_matrix : numpy.ndarray(2d)
            Global input features matrix stored as a numpy.ndarray(2d) of
            shape (1, n_features).
        """
        if global_features_matrix is None:
            self._global_features_matrix = None
        else:
            # Check global input features matrix
            if not isinstance(global_features_matrix, np.ndarray):
                raise RuntimeError('Global input features matrix must be '
                                   'provided as a numpy 2d array of shape '
                                   '(1, n_features).')
            elif len(global_features_matrix.shape) != 2:
                raise RuntimeError('Global input features matrix must be '
                                   'provided as a numpy 2d array of shape '
                                   '(1, n_features).')
            elif global_features_matrix.shape[0] != 1:
                raise RuntimeError('Global input features matrix must be '
                                   'provided as a numpy 2d array of shape '
                                   '(1, n_features).')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set global input features matrix
            self._global_features_matrix = \
                copy.deepcopy(global_features_matrix)
    # -------------------------------------------------------------------------    
    def get_global_features_matrix(self):
        """Get global input features matrix.
        
        Returns
        -------
        global_features_matrix : numpy.ndarray(2d)
            Global input features matrix stored as a numpy.ndarray(2d) of
            shape (1, n_features).
        """
        return copy.deepcopy(self._global_features_matrix)
    # -------------------------------------------------------------------------   
    def set_node_targets_matrix(self, node_targets_matrix):
        """Set node targets matrix.
        
        Parameters
        ----------
        node_targets_matrix : numpy.ndarray(2d)
            Nodes targets matrix stored as a numpy.ndarray(2d) of shape
            (n_nodes, n_targets).
        """
        if node_targets_matrix is None:
            self._node_targets_matrix = None
        else:
            # Check nodes targets matrix
            if not isinstance(node_targets_matrix, np.ndarray):
                raise RuntimeError('Nodes targets matrix must be provided '
                                   'as a numpy 2d array of shape '
                                   '(n_nodes, n_targets).')
            elif len(node_targets_matrix.shape) != 2:
                raise RuntimeError('Nodes targets matrix matrix must be '
                                   'provided as a numpy 2d array of shape '
                                   '(n_nodes, n_targets).')
            elif node_targets_matrix.shape[0] != self._n_node:
                raise RuntimeError(f'Nodes targets matrix shape '
                                   f'({node_targets_matrix.shape}) is not '
                                   f'compatible with number of nodes of '
                                   f'graph ({self._n_node}).')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set nodes targets matrix
            self._node_targets_matrix = copy.deepcopy(node_targets_matrix)
    # -------------------------------------------------------------------------    
    def get_node_targets_matrix(self):
        """Get node targets matrix.
        
        Returns
        -------
        node_targets_matrix : numpy.ndarray(2d)
            Nodes targets matrix stored as a numpy.ndarray(2d) of shape
            (n_nodes, n_targets).
        """
        return copy.deepcopy(self._node_targets_matrix)
    # -------------------------------------------------------------------------   
    def set_edge_targets_matrix(self, edge_targets_matrix):
        """Set edge targets matrix.
        
        Parameters
        ----------
        edge_targets_matrix : numpy.ndarray(2d)
            Edges targets matrix stored as a numpy.ndarray(2d) of shape
            (n_edges, n_targets).
        """
        if edge_targets_matrix is None:
            self._edge_targets_matrix = None
        else:
            # Check edges targets matrix
            if not isinstance(edge_targets_matrix, np.ndarray):
                raise RuntimeError('Edges targets matrix must be provided '
                                   'as a numpy 2d array of shape '
                                   '(n_edges, n_targets).')
            elif len(edge_targets_matrix.shape) != 2:
                raise RuntimeError('Edges targets matrix matrix must be '
                                   'provided as a numpy 2d array of shape '
                                   '(n_edges, n_targets).')
            elif (self._n_edge is not None
                  and (edge_targets_matrix.shape[0] != self._n_edge)):
                raise RuntimeError(f'Edges targets matrix shape '
                                   f'({edge_targets_matrix.shape}) is not '
                                   f'compatible with number of edges of '
                                   f'graph ({self._n_edge}).')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set edges targets matrix
            self._edge_targets_matrix = copy.deepcopy(edge_targets_matrix)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set unknown number of edges according with edges targets matrix
            if self._n_edge is None:
                self._n_edge = self.edge_targets_matrix.shape[0]
    # -------------------------------------------------------------------------    
    def get_edge_targets_matrix(self):
        """Get edge targets matrix.
        
        Returns
        -------
        edge_targets_matrix : numpy.ndarray(2d)
            Edges targets matrix stored as a numpy.ndarray(2d) of shape
            (n_edges, n_targets).
        """
        return copy.deepcopy(self._edge_targets_matrix)
    # -------------------------------------------------------------------------   
    def set_global_targets_matrix(self, global_targets_matrix):
        """Set global targets matrix.
        
        Parameters
        ----------
        global_targets_matrix : numpy.ndarray(2d)
            Global targets matrix stored as a numpy.ndarray(2d) of shape
            (1, n_targets).
        """
        if global_targets_matrix is None:
            self._global_targets_matrix = None
        else:
            # Check global targets matrix
            if not isinstance(global_targets_matrix, np.ndarray):
                raise RuntimeError('Global targets matrix must be provided '
                                   'as a numpy 2d array of shape '
                                   '(1, n_targets).')
            elif len(global_targets_matrix.shape) != 2:
                raise RuntimeError('Global targets matrix matrix must be '
                                   'provided as a numpy 2d array of shape '
                                   '(1, n_targets).')
            elif global_targets_matrix.shape[0] != 1:
                raise RuntimeError('Global targets matrix must be provided '
                                   'as a numpy 2d array of shape '
                                   '(1, n_targets).')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set global targets matrix
            self._global_targets_matrix = copy.deepcopy(global_targets_matrix)
    # -------------------------------------------------------------------------    
    def get_global_targets_matrix(self):
        """Get global targets matrix.
        
        Returns
        -------
        global_targets_matrix : numpy.ndarray(2d)
            Global targets matrix stored as a numpy.ndarray(2d) of shape
            (1, n_targets).
        """
        return copy.deepcopy(self._global_targets_matrix)
    # -------------------------------------------------------------------------
    def set_metadata(self, metadata):
        """Set metadata information.
        
        Parameters
        ----------
        metadata : dict
            Any metadata information stored as a dictionary.

        Note
        ----
        Metadata information is useful to store any additional information that
        is not related to the graph data itself but that is relevant for further
        processing or analysis, e,g., sample id, time step, etc. 

   
        `metadata` is a dictionary that won't be moved to the GPU when using
        ``.to(device)``. Avoid retrieving information from it inside your
        a model.
        """
        # Check metadata information
        if not isinstance(metadata, dict):
            raise RuntimeError('Metadata information must be provided as a '
                               'dictionary.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set metadata information
        self._metadata = metadata
    # -------------------------------------------------------------------------
    def add_metadata(self, key, value):
        """Add metadata information.
        
        Parameters
        ----------
        key : str
            Key of metadata information.
        value : Any
            Value of metadata information.
        """
        # Check metadata
        if not isinstance(self._metadata, dict):
            raise RuntimeError('Metadata information must be a dictionary.')
        # Check key
        if not isinstance(key, str):
            raise RuntimeError('Key of metadata information must be a string.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Add metadata information
        self._metadata[key] = value
    # -------------------------------------------------------------------------
    def get_metadata(self):
        """Get metadata information.
        
        Returns
        -------
        metadata : dict
            Metadata information stored as a dictionary.
        """
        return copy.deepcopy(self._metadata)
    def plot_graph(self, is_show_plot=False, is_save_plot=False,
                   save_directory=None, plot_name=None,
                   is_overwrite_file=False):
        """Generate plot of graph.
        
        Parameters
        ----------
        is_show_plot : bool, default=False
            Display plot of graph if True.
        is_save_plot : bool, default=False
            Save plot of graph. Plot is only saved if `save_directory` is
            provided and exists.
        save_directory : str, default=None
            Directory where plot of graph is stored.
        plot_name : str, default=None
            Filename of graph plot.
        is_overwrite_file : bool, default=False
            Overwrite plot of graph if True, otherwise generate
            generate non-existent file path by extending the original file path
            with an integer.
        """
        # Get nodes coordinates  
        nodes_coords = self._nodes_coords
        # Get edges
        edges_indexes = self._edges_indexes
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get number of dimensions
        n_dim = nodes_coords.shape[1]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Generate plot according with number of dimensions
        if n_dim == 2:
            # Generate 2D plot
            fig, ax = plt.subplots()
            # Plot nodes
            if nodes_coords is not None:
                ax.plot(nodes_coords[:, 0], nodes_coords[:, 1],
                        'o', color='#d62728', label='Nodes', zorder=10)
            # Plot edges
            if nodes_coords is not None and edges_indexes is not None:
                for (i, j) in edges_indexes:
                    ax.plot([nodes_coords[i, 0], nodes_coords[j, 0]],
                            [nodes_coords[i, 1], nodes_coords[j, 1]],
                            '-', color='#1d77b4', zorder=5)          
            ax.plot([], [], '-', color='#1d77b4', label='Edges', zorder=5)
        elif n_dim == 3:
            # Generate 3D plot
            fig, ax = plt.subplots(subplot_kw={'projection': '3d',
                                               'computed_zorder': False})
            # Hide grid lines
            ax.grid(False)
            # Plot nodes
            if nodes_coords is not None:
                ax.scatter(nodes_coords[:, 0], nodes_coords[:, 1],
                           nodes_coords[:, 2], color='#d62728', label='Nodes',
                           depthshade=False, zorder=10)
            # Plot edges
            if nodes_coords is not None and edges_indexes is not None:
                for (i, j) in edges_indexes:
                    ax.plot([nodes_coords[i, 0], nodes_coords[j, 0]],
                            [nodes_coords[i, 1], nodes_coords[j, 1]],
                            [nodes_coords[i, 2], nodes_coords[j, 2]],
                            '-', color='#1d77b4', zorder=5)
            ax.plot([], [], '-', color='#1d77b4', label='Edges', zorder=5)
        else:
            raise RuntimeError('Invalid number of dimensions.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set plot legend
        ax.legend(loc='center', ncol=3, numpoints=1, frameon=True,
                    fancybox=True, facecolor='inherit', edgecolor='inherit',
                    fontsize=10, framealpha=1.0,
                    bbox_to_anchor=(0, 1.05, 1.0, 0.1),
                    borderaxespad=0.1)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Display figure
        if is_show_plot:
            plt.show()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Save figure (pdf format)
        if is_save_plot and os.path.exists(str(save_directory)):
            # Set default figure name
            if plot_name is None:
                plot_name = 'plot_graph'
            # Set figure path
            fig_path = os.path.join(os.path.normpath(save_directory),
                                    plot_name) + '.pdf'
            if os.path.isfile(fig_path) and not is_overwrite_file:
                fig_path = new_file_path_with_int(fig_path)
            # Set figure size (inches)
            fig.set_figheight(3.6, forward=False)
            fig.set_figwidth(3.6, forward=False)
            # Save figure file
            fig.savefig(fig_path, transparent=False, dpi=300,
                        bbox_inches='tight')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Close plot
        plt.close(fig)
    # -------------------------------------------------------------------------
    @staticmethod
    def _get_edges_from_local_radius(nodes_coords, connect_radius):
        """Get edges between nodes that are within a given connectivity radius.
        
        Parameters
        ----------
        nodes_coords : numpy.ndarray(2d)
            Coordinates of nodes stored as a numpy.ndarray(2d) with shape
            (n_nodes, n_dim). Coordinates of i-th node are stored in
            nodes_coords[i, :].
        connect_radius : float
            Connectivity radius that sets the maximum distance between two
            nodes that leads to an edge.
        
        Returns
        -------
        edges_indexes : numpy.ndarray(2d)
            Edges indexes matrix stored as numpy.ndarray[int](2d) with shape
            (num_edges, 2), where the i-th edge is stored in
            edges_indexes[i, :] as (start_node_index, end_node_index).
        """
        # Initialize k-d tree
        kd_tree = scipy.spatial.KDTree(nodes_coords)
        # Find all edges between nodes that are at most within a given distance
        # between them
        edges_indexes = kd_tree.query_pairs(r=connect_radius, p=2.0,
                                            output_type='ndarray')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get undirected edges indexes
        edges_indexes = GraphData.get_undirected_unique_edges(edges_indexes) 
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return edges_indexes
    # -------------------------------------------------------------------------
    @staticmethod
    def get_edges_indexes_mesh(connected_nodes):
        """Convert set of mesh connected nodes to edges indexes matrix.
        
        It is assumed that nodes are labeled from 1 to n_nodes, such that
        node 1 and node n_nodes are associated with indexes 0 and n_nodes-1,
        respectively.
        
        Parameters
        ----------
        connected_nodes : tuple[tuple(2)]
            A set containing all pairs of nodes that are connected by any
            relevant mesh representation (e.g., finite element mesh). Each
            connection is stored a single time as a tuple(node[int], node[int])
            and is independent of the corresponding nodes storage order.
            
        Returns
        -------
        edges_indexes_mesh : numpy.ndarray(2d), default=None
            Edges stemming from any relevant mesh representation (e.g., finite
            element mesh) and that should be accounted for. Edges indexes
            matrix stored as numpy.ndarray[int](2d) with shape (n_edges, 2),
            where the i-th edge is stored in edges_indexes[i, :] as
            (start_node_index, end_node_index).
        """
        # Initialize mesh edges indexes matrix
        edges_indexes_mesh = np.zeros((len(connected_nodes), 2), dtype=int)        
        # Loop over mesh edges
        for i, edge in enumerate(connected_nodes):
            # Assemble edge indexes
            edges_indexes_mesh[i, :] = (edge[0] - 1, edge[1] - 1)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get unidirected unique edges indexes
        edges_indexes_mesh = \
            GraphData.get_undirected_unique_edges(edges_indexes_mesh)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return edges_indexes_mesh
    # -------------------------------------------------------------------------
    @staticmethod
    def get_undirected_unique_edges(edges_indexes):
        """Get set of undirected unique edges indexes.
        
        This function processes the given matrix of edges indexes and
        transforms all edges into undirected edges. In addition, it also
        removes any duplicated edges.
        
        Parameters
        ----------
        edges_indexes : numpy.ndarray(2d)
            Edges indexes matrix stored as numpy.ndarray[int](2d) with shape
            (n_edges, 2), where the i-th edge is stored in
            edges_indexes[i, :] as (start_node_index, end_node_index).
        
        Returns
        -------
        edges_indexes : numpy.ndarray(2d)
            Edges indexes matrix stored as numpy.ndarray[int](2d) with shape
            (n_edges, 2), where the i-th edge is stored in
            edges_indexes[i, :] as (start_node_index, end_node_index).
        """
        # Check provided edges indexes matrix
        GraphData._check_edges_indexes_matrix(edges_indexes)
        # Transforms all edges into undirected edges
        edges_indexes = \
            np.concatenate((edges_indexes, edges_indexes[:, ::-1]), axis=0)
        # Remove duplicated edges
        edges_indexes = np.unique(edges_indexes, axis=0)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return edges_indexes
    # -------------------------------------------------------------------------
    @staticmethod
    def _check_edges_indexes_matrix(edges_indexes):
        """Check if given edges indexes matrix is valid.
        
        Parameters
        ----------
        edges_indexes : numpy.ndarray(2d)
            Edges indexes matrix stored as numpy.ndarray[int](2d) with shape
            (n_edges, 2), where the i-th edge is stored in edges_indexes[i, :]
            as (start_node_index, end_node_index).
        """
        if not isinstance(edges_indexes, np.ndarray):
            raise RuntimeError('Edges indexes matrix is not a numpy.array.')
        elif edges_indexes.dtype != int:
            raise RuntimeError('Edges indexes matrix is not a numpy.array '
                               'of dtype int.')
        elif len(edges_indexes.shape) != 2:
            raise RuntimeError('Edges indexes matrix is not a numpy.array '
                               'of shape (n_edges, 2).')
        elif edges_indexes.shape[1] != 2:
            raise RuntimeError('Edges indexes matrix is not a numpy.array '
                               'of shape (n_edges, 2).')
    # -------------------------------------------------------------------------
    @staticmethod
    def move_time_dimension_axis(array, source_dim, dest_dim):
        """Move time dimension axis in temporal data array.
        
        Parameters
        ----------
        array : np.ndarray
            Temporal data array.
        source_dim : int
            Original index of time dimension.
        dest_dim : int
            Destination index of time dimension.
            
        Returns
        -------
        array : np.ndarray
            Reshaped temporal data array.
        """
        # Check temporal data array
        if not isinstance(array, np.ndarray):
            raise RuntimeError(f'Temporal data array must be numpy.ndarray, '
                               f'not {type(array)}.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Move temporal dimension axis
        array = np.moveaxis(array, source_dim, dest_dim)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return array