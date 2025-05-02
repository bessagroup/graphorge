"""Graph Neural Networks architectures.

Classes
-------
GraphIndependentNetwork(torch.nn.Module)
    Graph Independent Network.
GraphInteractionNetwork(torch_geometric.nn.MessagePassing)
    Graph Interaction Network.
 
Functions
---------
build_fnn
    Build multilayer feed-forward neural network.
"""
#
#                                                                       Modules
# =============================================================================
# Third-party
import torch
import torch_geometric.nn
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', 'Rui Barreira']
__status__ = 'Planning'
# =============================================================================
#
# =============================================================================
def build_fnn(input_size, output_size,
              output_activation=torch.nn.Identity(),
              hidden_layer_sizes=[],
              hidden_activation=torch.nn.Identity()):
    """Build multilayer feed-forward neural network.
    
    Parameters
    ----------
    input_size : int
        Number of neurons of input layer.
    output_size : int
        Number of neurons of output layer.
    output_activation : torch.nn.Module, default=torch.nn.Identity
        Output unit activation function. Defaults to identity (linear) unit
        activation function.
    hidden_layer_sizes : list[int], default=[]
        Number of neurons of hidden layers.
    hidden_activation : torch.nn.Module, default=torch.nn.Identity
        Hidden unit activation function. Defaults to identity (linear) unit
        activation function.

    Returns
    -------
    fnn : torch.nn.Sequential
        Multilayer feed-forward neural network.
    """
    # Check input and output size
    if int(input_size) < 1 or int(output_size) < 1:
        raise RuntimeError(f'Number of input ({int(input_size)}) and output '
                           f'({output_size}) features must be at least 1.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set number of neurons of each layer
    layer_sizes = []
    layer_sizes.append(input_size)
    layer_sizes += hidden_layer_sizes
    layer_sizes.append(output_size)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set number of layers of adaptive weights
    n_layer = len(layer_sizes) - 1
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set hidden and output layers unit activation functions
    if not callable(hidden_activation):
        raise RuntimeError('Hidden unit activation function must be derived '
                           'from torch.nn.Module class.')
    activation_functions = [hidden_activation for i in range(n_layer - 1)]
    if not callable(output_activation):
        raise RuntimeError('Output unit activation function must be derived '
                           'from torch.nn.Module class.')
    activation_functions.append(output_activation)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Create multilayer feed-forward neural network:
    # Initialize neural network
    fnn = torch.nn.Sequential()
    # Loop over neural network layers
    for i in range(n_layer):
        # Set layer linear transformation
        fnn.add_module("Layer-" + str(i),
                       torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1],
                                       bias=True))
        # Set layer unit activation function
        fnn.add_module("Activation-" + str(i), activation_functions[i])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return fnn
# =============================================================================
def build_rnn(input_size, hidden_layer_sizes, output_size, num_layers=None,
              rnn_cell='GRU', bias=True):
    """Build multilayer recurrent neural network.

    Parameters
    ----------
    input_size : int
        Number of neurons of input layer.
    hidden_layer_sizes : list[int]
        Number of neurons of hidden layers.
    output_size : int
        Number of neurons of output layer.
    num_layers : list[int], default=None
        Number of layers in each RNN module. Defaults to None.
    rnn_cell : str, default='GRU'
        RNN architecture cell.
    bias : bool, default=True
        Whether to use bias within the RNN cell.

    Returns
    -------
    rnn : torch.nn.Sequential
        Multilayer recurrent neural network with linear output layer.
    """
    # Check input and output size
    if int(input_size) < 1 or int(output_size) < 1:
        raise RuntimeError(f'Number of input ({int(input_size)}) and output '
                           f'({output_size}) features must be at least 1.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check cells
    if rnn_cell != 'GRU':
        raise RuntimeError(f'({rnn_cell}) is not a recognized RNN cell.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if hidden_layer_sizes is None:
        # This ensures at least one pass through an RNN, before the linear
        # output layer.
        hidden_layer_sizes = [output_size, ]
    if num_layers is None:
        num_layers = []
        for _ in range(len(hidden_layer_sizes)):
            num_layers.append(1)
    elif len(num_layers) != len(hidden_layer_sizes):
            raise RuntimeError(f'Expected same '
                               f'number of ''num_layers'' as'
                               f' ''hidden_layer_sizes''. Instead, got '
                               f'{len(num_layers)} and '
                               f'{len(hidden_layer_sizes)}.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set number of neurons of each layer
    layer_sizes = []
    layer_sizes.append(input_size)
    layer_sizes += hidden_layer_sizes
    layer_sizes.append(output_size)
    n_layer = len(layer_sizes) - 1
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Create multilayer recurrent neural network:
    rnn = torch.nn.Sequential()
    # Loop over neural network layers
    for idx in range(n_layer - 1):
        # Set layer linear transformation
        rnn.add_module(rnn_cell + '-' + str(idx),
                       torch.nn.GRU(input_size=layer_sizes[idx], 
                                    hidden_size=layer_sizes[idx+1],
                                    bias=bias, num_layers=num_layers[idx]))
        # Extracts 'output' from '(output, hidden_states)'. 'hidden_state' is 
        # not passed between RNN cells, and it are internal variable to each
        # RNN-cell.
        rnn.add_module('RNN-output' + str(idx),TorchRNNWrapper())
    # Linear output layer (last hidden_size -> output_size)
    rnn.add_module('Output-layer',torch.nn.Linear(in_features=layer_sizes[-2],
                                                  out_features=layer_sizes[-1],
                                                  bias=bias))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return rnn
# =============================================================================
class TorchRNNWrapper(torch.nn.Module):
    """TorchRNNWrapper.
    
    Extracts 'output' features from the tuple '(output, hidden_state)' returned
    by the recurrent neural network architectures implemented in Pytorch.
    'hidden_state' is discarded.

    Methods
    -------
    forward(self, x)
        Forward propagation.
    """
    def forward(self, x):
        """Forward propagation.
        
        Parameters
        ----------
        x : Tuple
            Tuple with (output, hidden_state) as returned by the recurrent 
            neural network architectures implemented in Pytorch.
            
        Returns
        -------
        output : {torch.Tensor, None}
            First element of the tuple, corresponding to the output of the 
            recurrent neural network. 
        """
        output, _ = x
        return output
# =============================================================================
class GraphIndependentNetwork(torch.nn.Module):
    """Graph Independent Network.
    
    A Graph Network block with (1) distinct update functions for node, edge and
    global features implemented as multilayer feed-forward neural networks with
    layer normalization and (2) no aggregation functions, i.e., independent
    node, edges and global blocks.
    
    Attributes
    ----------
    _node_fn : torch.nn.Sequential
        Node update function.
    _n_node_in : int
        Number of node input features.
    _n_node_out : int
        Number of node output features.
    _edge_fn : torch.nn.Sequential
        Edge update function.
    _n_edge_in : int
        Number of edge input features.
    _n_edge_out : int
        Number of edge output features.
    _global_fn : torch.nn.Sequential
        Global update function.
    _n_global_in : int
        Number of global input features.
    _n_global_out : int
        Number of global output features.
    _n_time_node : int
        Number of discrete time steps of nodal features.
        If greater than 0, then nodal input features include a time
        dimension, and message passing layers are RNNs.
    _n_time_edge : int
        Number of discrete time steps of edge features.
        If greater than 0, then edge input features include a time 
        dimension, and message passing layers are RNNs.
    _n_time_global : int
        Number of discrete time steps of global features.
        If greater than 0, then global input features include a time
        dimension, and message passing layers are RNNs.
    _norm_layer : str, default=None
            Normalization layer to node, edge and global update functions.
    _is_skip_unset_update : bool
        If True, then return features input matrix when the corresponding
        update function has not been setup, otherwise return None.

    Methods
    -------
    forward(self, node_features_in=None, edge_features_in=None, \
            global_features_in)
        Forward propagation.
    """
    def __init__(self, n_hidden_layers, hidden_layer_size,
                 n_node_in=0, n_node_out=0,
                 n_edge_in=0, n_edge_out=0,
                 n_global_in=0, n_global_out=0,
                 n_time_node=0, n_time_edge=0, n_time_global=0,
                 node_hidden_activation=torch.nn.Identity(),
                 node_output_activation=torch.nn.Identity(),
                 edge_hidden_activation=torch.nn.Identity(),
                 edge_output_activation=torch.nn.Identity(),
                 global_hidden_activation=torch.nn.Identity(),
                 global_output_activation=torch.nn.Identity(),
                 norm_layer=None,
                 is_skip_unset_update=False):
        """Constructor.
        
        Parameters
        ----------
        n_hidden_layers : int
            Number of hidden layers of multilayer feed-forward neural network
            update functions.
        hidden_layer_size : int
            Number of neurons of hidden layers of multilayer feed-forward
            neural network update functions.
        n_node_in : int, default=0
            Number of node input features. Must be greater than zero to setup
            node update function.
        n_node_out : int, default=0
            Number of node output features. Must be greater than zero to setup
            node update function.
        n_edge_in : int, default=0
            Number of edge input features. Must be greater than zero to setup
            edge update function.
        n_edge_out : int, default=0
            Number of edge output features. Must be greater than zero to setup
            edge update function.
        n_global_in : int, default=0
            Number of global input features. Must be greater than zero to setup
            global update function.
        n_global_out : int, default=0
            Number of global output features. Must be greater than zero to
            setup global update function.
        node_hidden_activation : torch.nn.Module, default=torch.nn.Identity
            Hidden unit activation function of node update function (multilayer
            feed-forward neural network). Defaults to identity (linear) unit
            activation function.
        node_output_activation : torch.nn.Module, default=torch.nn.Identity
            Output unit activation function of node update function (multilayer
            feed-forward neural network). Defaults to identity (linear) unit
            activation function.
        edge_hidden_activation : torch.nn.Module, default=torch.nn.Identity
            Hidden unit activation function of edge update function (multilayer
            feed-forward neural network). Defaults to identity (linear) unit
            activation function.
        edge_output_activation : torch.nn.Module, default=torch.nn.Identity
            Output unit activation function of edge update function (multilayer
            feed-forward neural network). Defaults to identity (linear) unit
            activation function.
        global_hidden_activation : torch.nn.Module, default=torch.nn.Identity
            Hidden unit activation function of global update function
            (multilayer feed-forward neural network). Defaults to identity
            (linear) unit activation function.
        global_output_activation : torch.nn.Module, default=torch.nn.Identity
            Output unit activation function of global update function
            (multilayer feed-forward neural network). Defaults to identity
            (linear) unit activation function.
        n_time_node : int, default=0
            Number of discrete time steps of nodal features.
            If greater than 0, then nodal input features include a time 
            dimension and message passing layers are RNNs.
        n_time_edge : int, default=0
            Number of discrete time steps of edge features.
            If greater than 0, then edge input features include a time
            dimension and message passing layers are RNNs.
        n_time_global : int, default=0
            Number of discrete time steps of global features.
            If greater than 0, then global input features include a time
            dimension and message passing layers are RNNs.
        norm_layer : str, default=None
            Add normalization layer to node, edge and global update functions.
        is_skip_unset_update : bool, default=False
            If True, then return features input matrix when the corresponding
            update function has not been setup, otherwise return None. Ignored
            if update function is setup.
        """
        # Initialize Graph Network block from base class
        super(GraphIndependentNetwork, self).__init__()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set number of features
        self._n_node_in = int(n_node_in)
        self._n_node_out = int(n_node_out)
        self._n_edge_in = int(n_edge_in)
        self._n_edge_out = int(n_edge_out)
        self._n_global_in = int(n_global_in)
        self._n_global_out = int(n_global_out)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set input with time dimension
        if n_time_node != 0 and n_time_edge !=0 and n_time_node != n_time_edge:
            raise RuntimeError(f'Number of time steps must match across '
                               f'nodal and edge features. Instead, got '
                               f'n_time_node=({n_time_node}) and '
                               f'n_time_edge=({n_time_edge})')
        if n_time_node != 0 and n_time_global !=0 and \
                n_time_node != n_time_global:
            raise RuntimeError(f'Number of time steps must match across '
                               f'nodal and global features. Instead, got '
                               f'n_time_global=({n_time_global}) and '
                               f'n_time_node=({n_time_node})')
        if n_time_edge != 0 and n_time_global !=0 and \
                n_time_edge != n_time_global:
            raise RuntimeError(f'Number of time steps must match across '
                               f'edge and global features. Instead, got '
                               f'n_time_global=({n_time_global}) and '
                               f'n_time_edge=({n_time_edge})')
        self._n_time_node = int(n_time_node)
        self._n_time_edge = int(n_time_edge)
        self._n_time_global = int(n_time_global)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set normalization layer
        self._norm_layer = norm_layer
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set node update function as multilayer feed-forward neural network
        # or recurrent neural network
        # with layer normalization
        if self._n_node_in > 0 and self._n_node_out > 0:
            # Set node update function
            self._node_fn = torch.nn.Sequential()
            if self._n_time_node > 0:
                rnn = build_rnn(input_size=self._n_node_in,
                                output_size=self._n_node_out,
                                hidden_layer_sizes=n_hidden_layers*[
                                    hidden_layer_size, ],
                                rnn_cell='GRU',
                                bias=True)
                self._node_fn.add_module('RNN', rnn)
            else:
                # Build multilayer feed-forward neural network
                fnn = build_fnn(
                    input_size=self._n_node_in,
                    output_size=self._n_node_out,
                    output_activation=node_output_activation,
                    hidden_layer_sizes=n_hidden_layers*[hidden_layer_size,],
                    hidden_activation=node_hidden_activation)
                self._node_fn.add_module('FNN', fnn)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Add normalization layer (per-feature) to node update function
            if norm_layer:
                if norm_layer == 'layernorm':
                    _norm_layer = torch.nn.LayerNorm(
                        normalized_shape=self._n_node_out,
                        elementwise_affine=True)
                elif norm_layer == 'batchnorm':
                    _norm_layer = torch.nn.BatchNorm1d(
                        num_features=self._n_node_out, affine=True)
                else:
                    raise RuntimeError('Unknown normalization layer '
                                       f'({norm_layer}).')
                self._node_fn.add_module(norm_layer, _norm_layer)
        else:
            self._node_fn = None
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set edge update function as multilayer feed-forward neural network
        # with layer normalization:
        if self._n_edge_in > 0 and self._n_edge_out > 0:
            # Set edge update function
            self._edge_fn = torch.nn.Sequential()
            if self._n_time_edge > 0:
                rnn = build_rnn(input_size=self._n_edge_in,
                                output_size=self._n_edge_out,
                                hidden_layer_sizes=n_hidden_layers*[
                                    hidden_layer_size, ],
                                rnn_cell='GRU',
                                bias=True)
                self._edge_fn.add_module('RNN', rnn)
            else:
                # Build multilayer feed-forward neural network
                fnn = build_fnn(
                    input_size=self._n_edge_in,
                    output_size=self._n_edge_out,
                    output_activation=edge_output_activation,
                    hidden_layer_sizes=n_hidden_layers*[hidden_layer_size,],
                    hidden_activation=edge_hidden_activation)
                self._edge_fn.add_module('FNN', fnn)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Add normalization layer (per-feature) to edge update function
            if norm_layer:
                if norm_layer == 'layernorm':
                    _norm_layer = torch.nn.LayerNorm(
                        normalized_shape=self._n_edge_out,
                        elementwise_affine=True)
                elif norm_layer == 'batchnorm':
                    _norm_layer = torch.nn.BatchNorm1d(
                        num_features=self._n_edge_out, affine=True)
                else:
                    raise RuntimeError('Unknown normalization layer '
                                       f'({norm_layer}).')
                self._edge_fn.add_module(norm_layer, _norm_layer)
        else:
            self._edge_fn = None        
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set global update function as multilayer feed-forward neural network
        # with layer normalization:
        if self._n_global_in > 0 and self._n_global_out > 0:
            # Set global update function
            self._global_fn = torch.nn.Sequential()
            if self._n_time_global > 0:
                rnn = build_rnn(input_size=self._n_global_in,
                                output_size=self._n_global_out,
                                hidden_layer_sizes=n_hidden_layers*[
                                    hidden_layer_size, ],
                                bias=True)
                self._global_fn.add_module('RNN', rnn)
            else:
                # Build multilayer feed-forward neural network
                fnn = build_fnn(
                    input_size=self._n_global_in,
                    output_size=self._n_global_out,
                    output_activation=global_output_activation,
                    hidden_layer_sizes=n_hidden_layers*[hidden_layer_size,],
                    hidden_activation=global_hidden_activation)
                self._global_fn.add_module('FNN', fnn)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Add normalization layer (per-element) to global update function
            if norm_layer:
                if self._n_global_in < 2:
                    raise RuntimeError(f'Number of global features '
                                       f'({self._n_global_in}) must be '
                                       f'greater than 1 to compute standard '
                                       f'deviation in the corresponding '
                                       f'update function normalization '
                                       f'layer.')
                else:
                    if norm_layer == 'layernorm':
                        _norm_layer = torch.nn.LayerNorm(
                            normalized_shape=self._n_global_out,
                            elementwise_affine=True)
                    elif norm_layer == 'batchnorm':
                        _norm_layer = torch.nn.BatchNorm1d(
                            num_features=self._n_global_out, affine=True)
                    else:
                        raise RuntimeError('Unknown normalization layer '
                                        f'({norm_layer}).')
                    self._global_fn.add_module(norm_layer, _norm_layer)
        else:
            self._global_fn = None
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check update functions
        if (self._node_fn is None and self._edge_fn is None
            and self._global_fn is None):
            raise RuntimeError('Graph Independent Network was initialized '
                               'without setting up any node, edge or global '
                               'update function. Set positive number of '
                               'features for at least the node, edge or '
                               'global update function.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set flag to handle unset update function output
        self._is_skip_unset_update = is_skip_unset_update
    # -------------------------------------------------------------------------
    def forward(self, node_features_in=None, edge_features_in=None,
                global_features_in=None, batch_vector=None):
        """Forward propagation.
        
        Parameters
        ----------
        node_features_in : torch.Tensor, default=None
            Nodes features input matrix stored as a torch.Tensor(2d) of shape
            (n_nodes, n_features). Ignored if node update function is not
            setup.
        edge_features_in : torch.Tensor, default=None
            Edges features input matrix stored as a torch.Tensor(2d) of shape
            (n_edges, n_features). Ignored if edge update function is not
            setup.
        global_features_in : torch.Tensor, default=None
            Global features input matrix stored as a torch.Tensor(2d) of shape
            (1, n_features). Ignored if global update function is not setup.
        batch_vector : torch.Tensor, default=None
            Batch vector stored as torch.Tensor(1d) of shape (n_nodes,),
            assigning each node to a specific batch subgraph. Required to
            process a graph holding multiple isolated subgraphs when batch
            size is greater than 1.
            
        Returns
        -------
        node_features_out : {torch.Tensor, None}
            Nodes features output matrix stored as a torch.Tensor(2d) of shape
            (n_nodes, n_features).
        edge_features_out : {torch.Tensor, None}
            Edges features output matrix stored as a torch.Tensor(2d) of shape
            (n_edges, n_features).
        global_features_out : {torch.Tensor, None}
            Global features output matrix stored as a torch.Tensor(2d) of shape
            (1, n_features).
        """
        # Check number of nodes and nodes features
        if self._node_fn is not None:
            if not isinstance(node_features_in, torch.Tensor):
                raise RuntimeError('Nodes features input matrix is not a '
                                   'torch.Tensor.')
            elif self._norm_layer and node_features_in.shape[0] < 2:
                raise RuntimeError(f'Number of nodes '
                                   f'({node_features_in.shape[0]}) must be '
                                   f'greater than 1 to compute standard '
                                   f'deviation in the corresponding update '
                                   f'function normalization layer.')
            elif self._n_time_node > 0 and \
                node_features_in.shape[1] != \
                    self._n_node_in*self._n_time_node:
                raise RuntimeError(f'Mismatch of number of node features of '
                                   f'model '
                                   f'({self._n_node_in*self._n_time_node}) '
                                   f'and nodes input features matrix '
                                   f'({node_features_in.shape[1]}).')
            elif self._n_time_node == 0 and \
                node_features_in.shape[1] != self._n_node_in:
                raise RuntimeError(f'Mismatch of number of node features of '
                                   f'model ({self._n_node_in}) and nodes '
                                   f'input features matrix '
                                   f'({node_features_in.shape[1]}).')
        # Check number of edges and edges features
        if self._edge_fn is not None:
            if not isinstance(edge_features_in, torch.Tensor):
                raise RuntimeError('Edges features input matrix is not a '
                                   'torch.Tensor.')
            elif self._norm_layer and edge_features_in.shape[0] < 2:
                raise RuntimeError(f'Number of edges '
                                   f'({edge_features_in.shape[0]}) must be '
                                   f'greater than 1 to compute standard '
                                   f'deviation in the corresponding update '
                                   f'function normalization layer.')
            elif self._n_time_edge > 0 and \
                edge_features_in.shape[1] != self._n_edge_in*self._n_time_edge:
                raise RuntimeError(f'Mismatch of number of edge features of '
                                   f'model '
                                   f'({self._n_edge_in*self._n_time_edge}) '
                                   f'and edges input features matrix '
                                   f'({edge_features_in.shape[1]}).')
            elif self._n_time_edge == 0 and \
                edge_features_in.shape[1] != self._n_edge_in:
                raise RuntimeError(f'Mismatch of number of edge features of '
                                   f'model ({self._n_edge_in}) and edges '
                                   f'input features matrix '
                                   f'({edge_features_in.shape[1]}).')
        # Check number global features
        if self._global_fn is not None:
            if not isinstance(global_features_in, torch.Tensor):
                raise RuntimeError('Global features input matrix is not a '
                                   'torch.Tensor.')
            elif self._n_time_global > 0 and \
                global_features_in.shape[1] != \
                    self._n_global_in*self._n_time_global:
                raise RuntimeError(f'Mismatch of number of global features of '
                                   f'model ('
                                   f'{self._n_global_in*self._n_time_global}) '
                                   f'and global input features matrix '
                                   f'({global_features_in.shape[1]}).')
            elif self._n_time_global == 0 and \
                global_features_in.shape[1] != self._n_global_in:
                raise RuntimeError(f'Mismatch of number of global features of '
                                   f'model ({self._n_global_in}) and global '
                                   f'input features matrix '
                                   f'({global_features_in.shape[1]}).')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Forward propagation: Node update function
        node_features_out = None
        if self._node_fn is not None:
            if self._n_time_node > 0:
                batch_size_node = node_features_in.shape[0]
                # If we have time series data, reshape to a 3d tensor:
                # (n_time_node, batch_size, n_node_in)
                node_features_in = \
                    node_features_in.view( \
                        self._n_time_node, batch_size_node, -1)
                # Compute node update 
                node_features_out = self._node_fn(node_features_in)
                # If we have time series data, reshape back to 2d tensor:
                # (batch_size, n_node_in * n_time_node)
                node_features_out = node_features_out.view(batch_size_node, -1)
            else:
                node_features_out = self._node_fn(node_features_in)
        else:
            if self._is_skip_unset_update:
                node_features_out = node_features_in       
        # Forward propagation: Edge update function
        edge_features_out = None
        if self._edge_fn is not None:
            # If we have time series data, reshape to a 3d tensor:
            # (n_time_edge, batch_size, n_edge_in)
            if self._n_time_edge > 0:
                batch_size_edge = edge_features_in.shape[0]
                edge_features_in = \
                    edge_features_in.view(
                        self._n_time_edge, batch_size_edge, -1)
                # Compute edge update
                edge_features_out = self._edge_fn(edge_features_in)
                # If we have time series data, reshape back to 2d tensor:
                # (batch_size, n_edge_in * n_time_edge)
                edge_features_out = edge_features_out.view(batch_size_edge,-1)
            else:
                # Compute edge update
                edge_features_out = self._edge_fn(edge_features_in)
        else:
            if self._is_skip_unset_update:
                edge_features_out = edge_features_in
        # Forward propagation: Global update function
        global_features_out = None
        if self._global_fn is not None:
            # If we have time series data, reshape to a 3d tensor:
            # (n_time_global, batch_size, n_global_in)
            if self._n_time_global > 0:
                batch_size_global = global_features_in.shape[0]
                global_features_in = \
                    global_features_in.view(
                        self._n_time_global, batch_size_global, -1)
                # Compute global update
                global_features_out = self._global_fn(global_features_in)
                # If we have time series data, reshape back to 2d tensor:
                # (batch_size, n_global_in * n_time_global)
                global_features_out = \
                    global_features_out.view(batch_size_global, -1)
            else:
                # Compute global update
                global_features_out = self._global_fn(global_features_in)
        else:
            if self._is_skip_unset_update:
                global_features_out = global_features_in
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return node_features_out, edge_features_out, global_features_out
# =============================================================================
class GraphInteractionNetwork(torch_geometric.nn.MessagePassing):
    """Graph Interaction Network.
    
    A Graph Network block with (1) distinct update functions for node, edge and
    global features implemented as multilayer feed-forward or recurrent 
    neural networks with layer normalization and (2) edge-to-node and 
    node-to-global aggregation functions.
    
    Attributes
    ----------
    _node_fn : torch.nn.Sequential
        Node update function.
    _n_node_in : int
        Number of node input features.
    _n_node_out : int
        Number of node output features.
    _edge_fn : torch.nn.Sequential
        Edge update function.
    _n_edge_in : int
        Number of edge input features.
    _n_edge_out : int
        Number of edge input features.
    _global_fn : torch.nn.Sequential
        Global update function.
    _n_global_in : int
        Number of global input features.
    _n_global_out : int
        Number of global output features.
    _n_time_node : int
        Number of discrete time steps of nodal features.
        If greater than 0, then nodal input features include a time
        dimension, and message passing layers are RNNs.
    _n_time_edge : int
        Number of discrete time steps of edge features.
        If greater than 0, then edge input features include a time 
        dimension, and message passing layers are RNNs.
    _n_time_global : int
        Number of discrete time steps of global features.
        If greater than 0, then global input features include a time
        dimension, and message passing layers are RNNs.
    _norm_layer : str, default=None
            Normalization layer to node, edge and global update functions.
        
    Methods
    -------
    forward(self, edges_indexes, node_features_in=None, edge_features_in=None)
        Forward propagation.
    message(self, node_features_in_i, node_features_in_j, \
            edge_features_in=None)
        Builds messages to node i from each edge (j, i) (edge update).
    update(self, node_features_in_aggr, node_features_in=None)
        Update node features.
    """
    def __init__(self, n_node_out, n_edge_out, n_hidden_layers,
                 hidden_layer_size, n_node_in=0, n_edge_in=0, n_global_in=0,
                 n_global_out=0, n_time_node=0, n_time_edge=0, n_time_global=0,
                 edge_to_node_aggr='add', node_to_global_aggr='add',
                 node_hidden_activation=torch.nn.Identity(),
                 node_output_activation=torch.nn.Identity(),
                 edge_hidden_activation=torch.nn.Identity(),
                 edge_output_activation=torch.nn.Identity(),
                 global_hidden_activation=torch.nn.Identity(),
                 global_output_activation=torch.nn.Identity(),
                 norm_layer=None):
        """Constructor.
        
        Parameters
        ----------
        n_node_out : int
            Number of node output features.
        n_edge_out : int
            Number of edge output features.
        n_hidden_layers : int
            Number of hidden layers of multilayer feed-forward neural network
            update functions.
        hidden_layer_size : int
            Number of neurons of hidden layers of multilayer feed-forward
            neural network update functions.
        n_node_in : int, default=0
            Number of node input features.
        n_edge_in : int, default=0
            Number of edge input features.
        n_global_in : int, default=0
            Number of global input features.
        n_global_out : int, default=0
            Number of global output features.
        edge_to_node_aggr : {'add',}, default='add'
            Edge-to-node aggregation scheme.
        node_to_global_aggr : {'add', 'mean'}, default='add'
            Node-to-global aggregation scheme.
        node_hidden_activation : torch.nn.Module, default=torch.nn.Identity
            Hidden unit activation function of node update function (multilayer
            feed-forward neural network). Defaults to identity (linear) unit
            activation function.
        node_output_activation : torch.nn.Module, default=torch.nn.Identity
            Output unit activation function of node update function (multilayer
            feed-forward neural network). Defaults to identity (linear) unit
            activation function.
        edge_hidden_activation : torch.nn.Module, default=torch.nn.Identity
            Hidden unit activation function of edge update function (multilayer
            feed-forward neural network). Defaults to identity (linear) unit
            activation function.
        edge_output_activation : torch.nn.Module, default=torch.nn.Identity
            Output unit activation function of edge update function (multilayer
            feed-forward neural network). Defaults to identity (linear) unit
            activation function.
        global_hidden_activation : torch.nn.Module, default=torch.nn.Identity
            Hidden unit activation function of global update function
            (multilayer feed-forward neural network). Defaults to identity
            (linear) unit activation function.
        global_output_activation : torch.nn.Module, default=torch.nn.Identity
            Output unit activation function of global update function
            (multilayer feed-forward neural network). Defaults to identity
            (linear) unit activation function.
        n_time_node : int, default=0
            Number of discrete time steps of nodal features.
            If greater than 0, then nodal input features include a time 
            dimension and message passing layers are RNNs.
        n_time_edge : int, default=0
            Number of discrete time steps of edge features.
            If greater than 0, then edge input features include a time
            dimension and message passing layers are RNNs.
        n_time_global : int, default=0
            Number of discrete time steps of global features.
            If greater than 0, then global input features include a time
            dimension and message passing layers are RNNs.
        norm_layer : str, default=None
            Add normalization layer to node, edge and global update functions.
        """
        # Set aggregation scheme
        if edge_to_node_aggr == 'add':
            aggregation = torch_geometric.nn.aggr.SumAggregation()
        else:
            raise RuntimeError('Unknown aggregation scheme.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set node-to-global aggregation scheme
        if node_to_global_aggr in ('add', 'mean'):
            self.node_to_global_aggr = node_to_global_aggr
        else:
            raise RuntimeError('Unknown node-to-global aggregation scheme.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set flow direction of message passing
        flow = 'source_to_target'
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize Graph Network block from base class
        super(GraphInteractionNetwork, self).__init__(aggr=aggregation,
                                                      flow=flow)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set number of features
        self._n_node_in = int(n_node_in)
        self._n_node_out = int(n_node_out)
        self._n_edge_in = int(n_edge_in)
        self._n_edge_out = int(n_edge_out)
        self._n_global_in = int(n_global_in)
        self._n_global_out = int(n_global_out)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set input with time dimension
        if n_time_node != 0 and n_time_edge !=0 and n_time_node != n_time_edge:
            raise RuntimeError(f'Number of time steps must match across '
                               f'nodal and edge features. Instead, got '
                               f'n_time_node=({n_time_node}) and '
                               f'n_time_edge=({n_time_edge})')
        if n_time_node != 0 and n_time_global !=0 and \
                n_time_node != n_time_global:
            raise RuntimeError(f'Number of time steps must match across '
                               f'nodal and global features. Instead, got '
                               f'n_time_global=({n_time_global}) and '
                               f'n_time_node=({n_time_node})')
        if n_time_edge != 0 and n_time_global !=0 and \
                n_time_edge != n_time_global:
            raise RuntimeError(f'Number of time steps must match across '
                               f'edge and global features. Instead, got '
                               f'n_time_global=({n_time_global}) and '
                               f'n_time_edge=({n_time_edge})')
        self._n_time_node = int(n_time_node)
        self._n_time_edge = int(n_time_edge)
        self._n_time_global = int(n_time_global)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set normalization layer
        self._norm_layer = norm_layer
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check number of input features
        if (self._n_node_in < 1 and self._n_edge_in < 1
            and self._n_global_in < 1):
            raise RuntimeError(f'Impossible to setup model without node '
                               f'({self._n_node_in}), edge '
                               f'({self._n_edge_in}), or global '
                               f'({self._n_global_in}) input features.')
        # Check number of output features
        if (self._n_node_out < 1 or self._n_edge_out < 1):
            raise RuntimeError(f'Number of node ({self._n_node_out}) and '
                               f'edge ({self._n_edge_out}) output features '
                               f'must be greater than 0.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set node update function as multilayer feed-forward/recurrent neural
        # network with layer normalization:
        self._node_fn = torch.nn.Sequential()
        if self._n_time_node > 0:
            # Build recurrent neural network
            rnn = build_rnn(input_size=self._n_node_in + self._n_edge_out,
                            output_size=self._n_node_out,
                            hidden_layer_sizes=n_hidden_layers*[
                                hidden_layer_size, ],
                            rnn_cell='GRU',
                            bias=True)
            self._node_fn.add_module('RNN', rnn)
        else:
            if self._n_time_edge > 0:
                # Build recurrent neural network
                rnn = build_rnn(input_size=self._n_node_in + self._n_edge_out,
                                output_size=self._n_node_out,
                                hidden_layer_sizes=n_hidden_layers*[
                                    hidden_layer_size, ],
                                rnn_cell='GRU',
                                bias=True)
                self._node_fn.add_module('RNN', rnn)
            else:
                # Build multilayer feed-forward neural network
                fnn = build_fnn(
                        input_size=self._n_node_in + self._n_edge_out,
                        output_size=self._n_node_out,
                        output_activation=node_output_activation,
                        hidden_layer_sizes=n_hidden_layers*[
                            hidden_layer_size,],
                        hidden_activation=node_hidden_activation)
                # Set node update function
                self._node_fn.add_module('FNN', fnn)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Add normalization layer (per-feature) to node update function
        if norm_layer:
                if norm_layer == 'layernorm':
                    _norm_layer = torch.nn.LayerNorm(
                        normalized_shape=self._n_node_out,
                        elementwise_affine=True)
                elif norm_layer == 'batchnorm':
                    _norm_layer = torch.nn.BatchNorm1d(
                        num_features=self._n_node_out, affine=True)
                else:
                    raise RuntimeError('Unknown normalization layer '
                                       f'({norm_layer}).')
                self._node_fn.add_module(norm_layer, _norm_layer)    
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set edge update function as multilayer feed-forward or recurrent
        # neural network with layer normalization:
        # Set edge update function
        self._edge_fn = torch.nn.Sequential()
        if self._n_time_edge > 0:
            # Build recurrent neural network
            # regardless if sef._n_time_node > 0 or not
            rnn = build_rnn(input_size=self._n_edge_in + 2*self._n_node_in,
                            output_size=self._n_edge_out,
                            hidden_layer_sizes=n_hidden_layers*[
                                hidden_layer_size,],
                            rnn_cell='GRU',
                            bias=True)
            self._edge_fn.add_module('RNN', rnn)
        else:
            if self._n_time_node > 0 or self._n_time_edge > 0:
                # Build recurrent neural network
                rnn = build_rnn(input_size=self._n_edge_in +
                                    2*self._n_node_in,
                                output_size=self._n_edge_out,
                                hidden_layer_sizes=n_hidden_layers*[
                                    hidden_layer_size,],
                                rnn_cell='GRU',
                                bias=True)
                self._edge_fn.add_module('RNN', rnn)
            else:
                # Build multilayer feed-forward neural network
                fnn = build_fnn(
                        input_size=self._n_edge_in + 2*self._n_node_in,
                        output_size=self._n_edge_out,
                        output_activation=edge_output_activation,
                        hidden_layer_sizes=n_hidden_layers*[
                            hidden_layer_size,],
                        hidden_activation=edge_hidden_activation)
                # Set edge update function
                self._edge_fn.add_module('FNN', fnn)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Add normalization layer (per-feature) to edge update function
        if norm_layer:
                if norm_layer == 'layernorm':
                    _norm_layer = torch.nn.LayerNorm(
                        normalized_shape=self._n_edge_out,
                        elementwise_affine=True)
                elif norm_layer == 'batchnorm':
                    _norm_layer = torch.nn.BatchNorm1d(
                        num_features=self._n_edge_out, affine=True)
                else:
                    raise RuntimeError('Unknown normalization layer '
                                       f'({norm_layer}).')
                self._edge_fn.add_module(norm_layer, _norm_layer)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set global update function as multilayer feed-forward neural network
        # with layer normalization:
        if self._n_global_out > 0:
            # Set global update function
            self._global_fn = torch.nn.Sequential()
            if self._n_time_global > 0:
                # Build stacked recurrent neural network
                # regardless if sef._n_time_node > 0 or self._n_time edge > 0 
                # or not
                rnn = build_rnn(input_size=self._n_global_in + 
                                    self._n_node_out,
                                output_size=self._n_global_out,
                                hidden_layer_sizes=n_hidden_layers*[
                                    hidden_layer_size,],
                                rnn_cell='GRU',
                                bias=True)
                self._global_fn.add_module('RNN', rnn)
            else:
                if self._n_time_node > 0 or self._n_time_edge > 0:
                    # Build stacked recurrent neural network
                    rnn = build_rnn(input_size=self._n_global_in +
                                        self._n_node_out,
                                    output_size=self._n_global_out,
                                    hidden_layer_sizes=n_hidden_layers*[
                                        hidden_layer_size,],
                                    rnn_cell='GRU',
                                    bias=True)
                    self._global_fn.add_module('RNN', rnn)
                else:
                    # Build multilayer feed-forward neural network
                    fnn = build_fnn(
                        input_size=self._n_global_in+self._n_node_out,
                        output_size=self._n_global_out,
                        output_activation=global_output_activation,
                        hidden_layer_sizes=n_hidden_layers*[
                            hidden_layer_size,],
                        hidden_activation=global_hidden_activation)
                    self._global_fn.add_module('FNN', fnn)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Add normalization layer (per-element) to global update function
            if norm_layer:
                if self._n_global_in < 2:
                    raise RuntimeError(f'Number of global features '
                                       f'({self._n_global_in}) must be '
                                       f'greater than 1 to compute standard '
                                       f'deviation in the corresponding '
                                       f'update function normalization layer.')
                else:
                    if norm_layer == 'layernorm':
                        _norm_layer = torch.nn.LayerNorm(
                            normalized_shape=self._n_node_out,
                            elementwise_affine=True)
                    elif norm_layer == 'batchnorm':
                        _norm_layer = torch.nn.BatchNorm1d(
                            num_features=self._n_node_out, affine=True)
                    else:
                        raise RuntimeError('Unknown normalization layer '
                                        f'({norm_layer}).')
                    self._global_fn.add_module(norm_layer, _norm_layer)
        else:
            self._global_fn = None
    # -------------------------------------------------------------------------
    def forward(self, edges_indexes, node_features_in=None,
                edge_features_in=None, global_features_in=None,
                batch_vector=None):
        """Forward propagation.
        
        Parameters
        ----------
        edges_indexes : torch.Tensor
            Edges indexes matrix stored as torch.Tensor(2d) with shape
            (2, n_edges), where the i-th edge is stored in edges_indexes[:, i]
            as (start_node_index, end_node_index).
        node_features_in : torch.Tensor, default=None
            Nodes features input matrix stored as a torch.Tensor(2d) of shape
            (n_nodes, n_features). If None, the edge-to-node aggregation is
            only built up to the highest receiver node index according with
            edges_indexes. To preserve total number of nodes in edge-to-node
            aggregation, pass torch.empty(n_nodes, 0) instead of None.
        edge_features_in : torch.Tensor, default=None
            Edges features input matrix stored as a torch.Tensor(2d) of shape
            (n_edges, n_features).
        global_features_in : torch.Tensor, default=None
            Global features input matrix stored as a torch.Tensor(2d) of shape
            (1, n_features). Ignored if global update function is not setup.
        batch_vector : torch.Tensor, default=None
            Batch vector stored as torch.Tensor(1d) of shape (n_nodes,),
            assigning each node to a specific batch subgraph. Required to
            process a graph holding multiple isolated subgraphs when batch
            size is greater than 1.
        
        Returns
        -------
        node_features_out : torch.Tensor
            Nodes features output matrix stored as a torch.Tensor(2d) of shape
            (n_nodes, n_features).
        edge_features_out : torch.Tensor
            Edges features output matrix stored as a torch.Tensor(2d) of shape
            (n_edges, n_features).
        global_features_out : {torch.Tensor, None}
            Global features output matrix stored as a torch.Tensor(2d) of shape
            (1, n_features).
        """
        # Check input features matrices
        if node_features_in is None and edge_features_in is None:
            raise RuntimeError('Impossible to compute forward propagation of '
                               'model without node (None) and edge (None) '
                               'input features matrices.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check edges indexes
        if not isinstance(edges_indexes, torch.Tensor):
            raise RuntimeError('Edges indexes matrix is not a torch.Tensor.')
        elif len(edges_indexes.shape) != 2 or edges_indexes.shape[0] != 2:
            raise RuntimeError('Edges indexes matrix is not a torch.Tensor '
                               'of shape (2, n_edges).')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check number of nodes and nodes features
        if node_features_in is not None:
            if not isinstance(node_features_in, torch.Tensor):
                raise RuntimeError('Nodes features input matrix is not a '
                                   'torch.Tensor.')
            elif self._norm_layer and node_features_in.shape[0] < 2:
                raise RuntimeError(f'Number of nodes '
                                   f'({node_features_in.shape[0]}) must be '
                                   f'greater than 1 to compute standard '
                                   f'deviation in the corresponding update '
                                   f'functions normalization layer.')
            elif self._n_time_node > 0 and \
                node_features_in.shape[1] != self._n_node_in*self._n_time_node:
                raise RuntimeError(f'Mismatch of number of node features of '
                                   f'model '
                                   f'({self._n_node_in*self._n_time_node}) '
                                   f'and nodes input features matrix '
                                   f'({node_features_in.shape[1]}).')
            elif self._n_time_node == 0 and \
                node_features_in.shape[1] != self._n_node_in:
                raise RuntimeError(f'Mismatch of number of node features of '
                                   f'model ({self._n_node_in}) and nodes '
                                   f'input features matrix '
                                   f'({node_features_in.shape[1]}).')
        # Check number of edges and edges features
        if edge_features_in is not None:
            if not isinstance(edge_features_in, torch.Tensor):
                raise RuntimeError('Edges features input matrix is not a '
                                   'torch.Tensor.')
            elif self._norm_layer and edge_features_in.shape[0] < 2:
                raise RuntimeError(f'Number of edges '
                                   f'({edge_features_in.shape[0]}) must be '
                                   f'greater than 1 to compute standard '
                                   f'deviation in the corresponding update '
                                   f'function normalization layer.')
            elif edge_features_in.shape[0] != edges_indexes.shape[1]:
                raise RuntimeError(f'Mismatch of number of edges of graph '
                                   f'edges indexes ({edges_indexes.shape[1]}) '
                                   f'and edges input features matrix '
                                   f'({edge_features_in.shape[1]}).')
            elif self._n_time_edge > 0 and \
                edge_features_in.shape[1] != \
                    self._n_edge_in * self._n_time_edge:
                raise RuntimeError(f'Mismatch of number of edge features of '
                                   f'model '
                                   f'({self._n_edge_in*self._n_time_edge }) '
                                   f'and edges input features matrix '
                                   f'({edge_features_in.shape[1]}).')
            elif self._n_time_edge == 0 and \
                edge_features_in.shape[1] != self._n_edge_in:
                raise RuntimeError(f'Mismatch of number of edge features of '
                                   f'model ({self._n_edge_in}) and edges '
                                   f'input features matrix '
                                   f'({edge_features_in.shape[1]}).')
        # Check global features
        if global_features_in is not None:
            if not isinstance(global_features_in, torch.Tensor):
                raise RuntimeError('Global features input matrix is not a '
                                   'torch.Tensor.')
            elif self._n_time_global > 0 and \
                global_features_in.shape[1] != \
                    self._n_global_in*self._n_time_global:
                raise RuntimeError(f'Mismatch of number of global features of '
                                   f'model '
                                   f'({self._n_global_in*self._n_time_global}) '
                                   f'and global input features matrix '
                                   f'({global_features_in.shape[1]}).') 
            elif self._n_time_global == 0 and \
                global_features_in.shape[1] != self._n_global_in:
                raise RuntimeError(f'Mismatch of number of global features of '
                                   f'model ({self._n_global_in}) and global '
                                   f'input features matrix '
                                   f'({global_features_in.shape[1]}).') 
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Perform graph neural network message-passing step (message,
        # aggregation, update) and get updated node features.
        # Time series data are reshaped within the message and update methods.
        node_features_out = self.propagate(
            edge_index=edges_indexes, node_features_in=node_features_in,
            edge_features_in=edge_features_in)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get updated edge features
        edge_features_out = self._edge_features_out
        self._edge_features_out = None
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize updated global features
        global_features_out = None
        # Get update global features
        # Time series data are reshaped within the update_global method.
        if self._global_fn is not None:
            global_features_out = self.update_global(
                global_features_in=global_features_in,
                node_features_out=node_features_out,
                batch_vector=batch_vector)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return node_features_out, edge_features_out, global_features_out
    # -------------------------------------------------------------------------
    def message(self, node_features_in_i, node_features_in_j,
                edge_features_in=None):
        """Builds messages to node i from each edge (j, i) (edge update).
        
        Assumes that j is the source node and i is the receiver node (flow
        direction set as 'source_to_target'). For each edge (j, i), the
        update function input features result from concatenation of the edge
        features and the corresponding nodes features.
        
        The source and receiver node input features mappings based on the edges
        indexes matrix are built in the _collect() method of class
        torch_geometric.nn.MessagePassing.
        
        The edges features output matrix is passed as the input tensor to the
        aggregation operator (class torch.nn.aggr.Aggregation) set in the
        initialization of the torch_geometric.nn.MessagePassing class.

        It is called by the "propagate" method within the PyG backend.
        
        Parameters
        ----------
        node_features_in_i : torch.Tensor
            Source node input features for each edge stored as a
            torch.Tensor(2d) of shape (n_edges, n_features). Mapping is
            performed based on the edges indexes matrix.
        node_features_in_j : torch.Tensor
            Receiver node input features for each edge stored as a
            torch.Tensor(2d) of shape (n_edges, n_features). Mapping is
            performed based on the edges indexes matrix.
        edge_features_in : torch.Tensor, default=None
            Edges features input matrix stored as a torch.Tensor(2d) of shape
            (n_edges, n_features).
            
        Returns
        -------
        edge_features_out : torch.Tensor
            Edges features output matrix stored as a torch.Tensor(2d) of shape
            (n_edges, n_features).
        """
        # Check input features
        is_node_features_in = (node_features_in_i is not None
                               and node_features_in_j is not None)
        is_edge_features_in = edge_features_in is not None
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Concatenate available input features for each edge
        if is_node_features_in and is_edge_features_in:
            if self._n_time_node > 0 and self._n_time_edge == 0:
                edge_features_in = \
                    edge_features_in.repeat(1, self._n_time_node)
            elif self._n_time_edge > 0 and self._n_time_node == 0:
                node_features_in_i = \
                    node_features_in_i.repeat(1, self._n_time_edge)
                node_features_in_j = \
                    node_features_in_j.repeat(1, self._n_time_edge)
            # Concatenate nodes and edges input features
            edge_features_in_cat = \
                torch.cat([node_features_in_i, node_features_in_j,
                        edge_features_in], dim=-1)
        elif is_node_features_in and not is_edge_features_in:
            if self._n_time_edge > 0:
                node_features_in_i = \
                    node_features_in_i.repeat(1, self._n_time_edge)
                node_features_in_j = \
                    node_features_in_j.repeat(1, self._n_time_edge)
            # Concatenate nodes input features
            edge_features_in_cat = \
                torch.cat([node_features_in_i, node_features_in_j], dim=-1)
        elif is_edge_features_in:
            if self._n_time_node > 0:
                edge_features_in = \
                    edge_features_in.repeat(1,self._n_time_node)
            # Concatenate edges input features
            edge_features_in_cat = edge_features_in
        else:
            raise RuntimeError('Impossible to build edge update function '
                               'input features matrix without node (None) and '
                               'edge (None) input features matrices')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update edge features
        # If we have time series data, reshape to a 3d tensor:
        # (n_time_edge, batch_size, n_edge_in)
        # If edges feature time series data, then it must be of the same size
        # as the time series data at the nodes
        if self._n_time_edge > 0 or self._n_time_node > 0:
            n_time = max(self._n_time_node, self._n_time_edge)
            batch_size_edge = edge_features_in_cat.shape[0]
            edge_features_in_cat = \
                edge_features_in_cat.view(n_time, batch_size_edge, -1)
            # Compute global update
            edge_features_out = self._edge_fn(edge_features_in_cat)
            # If we have time series data, reshape back to 2d tensor:
            # (batch_size, n_edge_in * n_time_edge)
            edge_features_out = edge_features_out.view(batch_size_edge, -1)
        else:
            # Compute global update
            edge_features_out = self._edge_fn(edge_features_in_cat)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Store updated edges features
        self._edge_features_out = edge_features_out
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return edge_features_out
    # -------------------------------------------------------------------------
    def update(self, node_features_in_aggr, node_features_in=None):
        """Update node features.
        
        The nodes features input matrix resulting from message passing and
        aggregation is built in the aggregation operator (class
        torch.nn.aggr.Aggregation) set in the initialization of the
        torch_geometric.nn.MessagePassing class.

        It is called by the propagate method within the PyG backend.
        
        Parameters
        ----------
        node_features_in_aggr : torch.Tensor
            Nodes features input matrix resulting from message passing and
            edge-to-node aggregation, stored as a torch.Tensor(2d) of shape
            (n_nodes, n_features).
        node_features_in : torch.Tensor, default=None
            Nodes features input matrix stored as a torch.Tensor(2d) of shape
            (n_nodes, n_features).
              
        Returns
        -------
        node_features_out : torch.Tensor
            Nodes features output matrix stored as a torch.Tensor(2d) of shape
            (n_nodes, n_features).
        """        
        # Concatenate features for each node:
        # Set node features stemming from edge-to-node aggregation
        # It is not necessary to extend node_features_in_aggr along the last
        # dimension, as this was already done in the message() function, when 
        # aggregating edge-to-nodal features. 
        # Only node_features_in needs to be extended, for the case 
        # self._n_time_edge > 0 and self._n_time_node == 0, as 
        # node_features_in_aggr will already be extended to include a time 
        # dimension
        node_features_in_cat = node_features_in_aggr
        # Concatenate available node input features
        if node_features_in is not None:
            # Check number of nodes stemming from edge-to-node aggregation
            if node_features_in_aggr.shape[0] != node_features_in.shape[0]:
                raise RuntimeError(f'Mismatch between number of nodes '
                                   f'stemming from edge-to-node aggregation '
                                   f'({node_features_in_aggr.shape[0]}) '
                                   f'and nodes features input matrix '
                                   f'({node_features_in.shape[0]}).')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if self._n_time_edge > 0 and self._n_time_node == 0:
                node_features_in = \
                    node_features_in.repeat(1,self._n_time_edge)
            node_features_in_cat = \
                torch.cat([node_features_in_cat, node_features_in], dim=-1)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update node features
        if self._n_time_node > 0 or self._n_time_edge > 0:
            n_time = max(self._n_time_node, self._n_time_edge)
            # If we have time series data, reshape to a 3d tensor:
            # (n_time_node, batch_size, n_node_in)
            batch_size_node = node_features_in_cat.shape[0]
            node_features_in_cat = \
                node_features_in_cat.view(n_time, batch_size_node, -1)
            # Compute global update
            node_features_out = self._node_fn(node_features_in_cat)
            # If we have time series data, reshape back to 2d tensor:
            # (batch_size, n_node_in * n_time_node)
            node_features_out = node_features_out.view(batch_size_node, -1)
        else:
            # Compute global update
            node_features_out = self._node_fn(node_features_in_cat)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return node_features_out
    # -------------------------------------------------------------------------
    def update_global(self, node_features_out, global_features_in=None,
                      batch_vector=None):
        """Update global features.
        
        Parameters
        ----------
        node_features_out : torch.Tensor
            Nodes features output matrix stored as a torch.Tensor(2d) of shape
            (n_nodes, n_features).
            If self._n_time_node > 0 or self._n_time_edge > 0, then 
            node_features_out has shape 
            (n_nodes, n_features*self._n_time_node/edge).
        global_features_in : torch.Tensor, default=None
            Global features input matrix stored as a torch.Tensor(2d) of shape
            (1, n_features).
        batch_vector : torch.Tensor, default=None
            Batch vector stored as torch.Tensor(1d) of shape (n_nodes,),
            assigning each node to a specific batch subgraph. Required to
            process a graph holding multiple isolated subgraphs when batch
            size is greater than 1.

        Returns
        -------
        global_features_out : torch.Tensor
            Global features output matrix stored as a torch.Tensor(2d) of shape
            (1, n_features).
        """
        # Perform node-to-global aggregation
        if self.node_to_global_aggr == 'add':
            node_features_in_aggr = torch_geometric.nn.global_add_pool(
                node_features_out, batch_vector)
        elif self.node_to_global_aggr == 'mean':
            node_features_in_aggr = torch_geometric.nn.global_mean_pool(
                node_features_out, batch_vector)
        else:
            raise RuntimeError('Unknown node-to-global aggregation scheme.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Concatenate global features:
        # Set global features stemming from node-to-global aggregation
        if self._n_time_global > 0 and \
            (self._n_time_node == 0 and self._n_time_edge == 0):
            global_features_in_cat = \
                node_features_in_aggr.repeat(1, self._n_time_global)
        else:
            global_features_in_cat = node_features_in_aggr
        # Concatenate available global input features
        if global_features_in is not None:
            # If (self._n_time_node > 0 or self._n_time_edge > 0) 
            # and self._n_time_global == 0:
            # then global features must be extended with a time dimension
            if (self._n_time_node > 0 or self._n_time_edge > 0) \
                and self._n_time_global == 0:
                n_time_in = max([self._n_time_node, self._n_time_edge])
                global_features_in = global_features_in.repeat(1, n_time_in)
            global_features_in_cat = \
                    torch.cat(
                        [global_features_in_cat, global_features_in],dim=-1)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update global features:
        # If we have time series data, reshape to a 3d tensor:
        # (n_time_global, batch_size, n_global_in)
        if self._n_time_global > 0 or self._n_time_node > 0 or \
            self._n_time_edge > 0:
            n_time = max([self._n_time_node, self._n_time_edge, 
                          self._n_time_global])
            batch_size_global = global_features_in_cat.shape[0]
            global_features_in_cat = \
                global_features_in_cat.view(n_time, batch_size_global, -1)
            # Compute global update
            global_features_out = self._global_fn(global_features_in_cat)
            # If we have time series data, reshape back to 2d tensor:
            # (batch_size, n_global_in * n_time_node)
            global_features_out = \
                global_features_out.view(batch_size_global, -1)
        else:
            global_features_out = self._global_fn(global_features_in_cat)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return global_features_out