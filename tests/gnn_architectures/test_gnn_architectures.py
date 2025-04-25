"""Test Graph Neural Networks architectures."""
#
#                                                                       Modules
# =============================================================================
# Third-party
import pytest
import torch
# Local
from src.graphorge.gnn_base_model.model.gnn_architectures import \
    build_fnn, build_rnn, GraphIndependentNetwork, GraphInteractionNetwork
# =============================================================================
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', 'Rui Barreira', ]
__status__ = 'Planning'
# =============================================================================
#
# =============================================================================
@pytest.mark.parametrize('input_size, output_size, output_activation,'
                         'hidden_layer_sizes, hidden_activation',
                         [(1, 1, torch.nn.Identity(), [], torch.nn.ReLU()),
                          (2, 3, torch.nn.Tanh(), [2, 3, 1],
                           torch.nn.LeakyReLU()),
                          ])
def test_build_fnn(input_size, output_size, output_activation,
                   hidden_layer_sizes, hidden_activation):
    """Test building of multilayer feed-forward neural network."""
    # Initialize errors
    errors = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build multilayer feed-forward neural network
    fnn = build_fnn(input_size, output_size, output_activation,
                    hidden_layer_sizes, hidden_activation)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get feed-forward neural network layers
    layers = [layer for layer in fnn.named_children() if 'Layer-' in layer[0]]
    activations = [layer for layer in fnn.named_modules(remove_duplicate=False)
                   if 'Activation-' in layer[0]]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check number of layers
    if len(layers) != len(hidden_layer_sizes) + 1:
        errors.append('Number of layers was not properly set.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over layers
    for i, layer in enumerate(layers):
        # Check input layer features              
        if i == 0:
            if layer[1].in_features != input_size:
                errors.append('Number of input features of input layer was '
                              'not properly set.')  
            if len(layers) == 1:
                if layer[1].out_features != output_size:
                    errors.append('Number of output features of input layer '
                                  'was not properly set.')
            else:
                if layer[1].out_features != hidden_layer_sizes[0]:
                    errors.append('Number of output features of input layer '
                                  'was not properly set.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check output layer features
        elif i == len(layers) - 1:
            if layer[1].in_features != hidden_layer_sizes[-1]:
                errors.append('Number of input features of output layer was '
                              'not properly set.')
            if layer[1].out_features != output_size:
                errors.append('Number of output features of output layer was '
                              'not properly set.')
            if not isinstance(activations[i][1], type(output_activation)):
                errors.append('Output unit activation function was not '
                              'properly set.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check hidden layer features
        else:
            if layer[1].in_features != hidden_layer_sizes[i - 1] \
                    or layer[1].out_features != hidden_layer_sizes[i]:
                errors.append('Number of input/output features of hidden '
                              'layer was not properly set.')
            if not isinstance(activations[i][1], type(hidden_activation)):
                errors.append('Hidden unit activation function was not '
                              'properly set.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    assert not errors, "Errors:\n{}".format("\n".join(errors))
# -----------------------------------------------------------------------------
@pytest.mark.parametrize('input_size, output_size, output_activation,'
                         'hidden_layer_sizes, hidden_activation',
                         [(1, 1, 'invalid_type', [], torch.nn.ReLU()),
                          (2, 3, torch.nn.Tanh, [2, 3, 1], 'invalid_type'),
                          ])
def test_build_fnn_invalid(input_size, output_size, output_activation,
                           hidden_layer_sizes, hidden_activation):
    """Test invalid activation functions."""
    with pytest.raises(RuntimeError):
        fnn = build_fnn(input_size, output_size, output_activation,
                        hidden_layer_sizes, hidden_activation)        
# -----------------------------------------------------------------------------
@pytest.mark.parametrize('input_size, hidden_layer_sizes, output_size,'
                         'num_layers, rnn_cell',
                         [(2, [128, 128, 256], 5, [2, 3, 1], 'GRU'),
                          (2, [], 5, [], 'GRU'),
                          ])
def test_build_rnn(input_size, hidden_layer_sizes, output_size, num_layers,
                   rnn_cell):
    """Test building of multilayer recurrent neural network."""
    # Initialize errors
    errors = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build multilayer feed-forward neural network
    rnn = build_rnn(input_size, hidden_layer_sizes, output_size, num_layers,
                    rnn_cell)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get recurrent neural network modules
    layer_name = rnn_cell + '-'
    layers = [layer for layer in rnn.named_children() if layer_name in layer[0]
              ]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check number of modules
    if len(layers) != len(hidden_layer_sizes):
        errors.append('Number of layers was not properly set.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over modules
    for i, layer in enumerate(layers):
        # Check input layer features              
        if i == 0:
            if layer[1].input_size != input_size:
                errors.append('Number of input features of input RNN layer '
                              'was not properly set.')  
            if layer[1].num_layers != num_layers[i]:
                errors.append('Number of layers of the RNN layer '
                              'was not properly set.') 
            if len(layers) == 1:
                if layer[1].hidden_size != output_size:
                    errors.append('Number of hidden features of input RNN '
                                  'layer was not properly set.')
            else:
                if layer[1].hidden_size != hidden_layer_sizes[0]:
                    errors.append('Number of hidden features of input RNN '
                                  'layer was not properly set.')
        else:
            if layer[1].input_size != hidden_layer_sizes[i-1]:
                errors.append('Number of input features of RNN layer '
                              'was not properly set.')
            if layer[1].hidden_size != hidden_layer_sizes[i]:
                    errors.append('Number of hidden features of RNN '
                                  'layer was not properly set.')
            if layer[1].num_layers != num_layers[i]:
                errors.append('Number of layers of the RNN layer '
                              'was not properly set.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check output layer features
    output_layer = [layer for layer in rnn.named_children() 
                     if 'Output-layer' in layer[0]][0]
    if len(hidden_layer_sizes) != 0:
        if output_layer[1].in_features != hidden_layer_sizes[-1]:
            errors.append('Number of input features of output layer was '
                            'not properly set.')
    if output_layer[1].out_features != output_size:
        errors.append('Number of output features of output layer was '
                            'not properly set.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    assert not errors, "Errors:\n{}".format("\n".join(errors))
# -----------------------------------------------------------------------------
@pytest.mark.parametrize('input_size, hidden_layer_sizes, output_size,'
                         'num_layers, rnn_cell',
                         [(2, [128, 128, 256], 5, [2, 3, 1], 'LSTM'),
                          (1, [128, 128, ], 5, [2, 3, 1], 'GRU'),
                          ])
def test_build_rnn_invalid(input_size, hidden_layer_sizes, output_size,
                           num_layers, rnn_cell):
    """Test invalid activation functions."""
    with pytest.raises(RuntimeError):
        rnn = build_rnn(input_size, hidden_layer_sizes, output_size,
                        num_layers, rnn_cell)
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
@pytest.mark.parametrize('n_node_in, n_node_out, n_edge_in, n_edge_out,'
                         'n_global_in, n_global_out,'
                         'n_hidden_layers, hidden_layer_size,'
                         'n_time_node, n_time_edge, n_time_global,'
                         'is_skip_unset_update, rnn_cell',
                         [(1, 5, 2, 3, 0, 0, 2, 4, 0, 0, 0, False, ''),
                          (3, 2, 1, 4, 2, 3, 1, 2, 0, 0, 0, False, ''),
                          (2, 4, 5, 4, 0, 0, 0, 2, 0, 0, 0, False, ''),
                          (0, 0, 5, 4, 3, 1, 0, 2, 0, 0, 0, False, ''),
                          (3, 2, 0, 0, 0, 0, 1, 2, 0, 0, 0, False, ''),
                          (1, 5, 2, 3, 0, 0, 2, 4, 0, 0, 0, True, ''),
                          (0, 0, 5, 4, 2, 3, 0, 2, 0, 0, 0, True, ''),
                          (3, 2, 0, 0, 0, 0, 1, 2, 0, 0, 0, True, ''),
                          (1, 5, 2, 3, 0, 0, 2, 4, 5, 5, 0, False, 'GRU'),
                          (3, 2, 1, 4, 2, 3, 1, 2, 2, 2, 2, False, 'GRU'),
                          (3, 2, 0, 0, 0, 0, 1, 2, 4, 0, 0, False, 'GRU'),
                          (1, 5, 2, 3, 0, 0, 2, 4, 2, 0, 0, True, 'GRU'),
                          (3, 2, 0, 0, 0, 0, 1, 2, 5, 0, 5, True, 'GRU'),
                          (0, 0, 0, 0, 5, 5, 3, 7, 0, 0, 5, True, 'GRU'),
                          (0, 0, 5, 4, 3, 1, 1, 2, 0, 5, 5, False, 'GRU'),
                          (2, 4, 5, 4, 0, 0, 1, 2, 3, 3, 0, False, 'GRU'),
                          (0, 0, 5, 4, 2, 3, 1, 2, 0, 7, 0, True, 'GRU'),
                          ])
def test_graph_independent_network_init(n_node_in, n_node_out, n_edge_in,
                                        n_edge_out, n_global_in, n_global_out,
                                        n_time_node, n_time_edge,
                                        n_time_global, n_hidden_layers,
                                        hidden_layer_size,
                                        is_skip_unset_update,
                                        rnn_cell):
    """Test Graph Independent Network constructor."""
    # Initialize errors
    errors = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build Graph Independent Network
    model = GraphIndependentNetwork(
        n_hidden_layers=n_hidden_layers, hidden_layer_size=hidden_layer_size,
        n_node_in=n_node_in, n_node_out=n_node_out,
        n_edge_in=n_edge_in, n_edge_out=n_edge_out,
        n_global_in=n_global_in, n_global_out=n_global_out,
        n_time_node=n_time_node, n_time_edge=n_time_edge,
        n_time_global=n_time_global,
        node_hidden_activation=torch.nn.ReLU(),
        node_output_activation=torch.nn.Identity(),
        edge_hidden_activation=torch.nn.ReLU(),
        edge_output_activation=torch.nn.Identity(),
        global_hidden_activation=torch.nn.ReLU(),
        global_output_activation=torch.nn.Identity(),
        is_skip_unset_update=is_skip_unset_update)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set Graph Independent Network update functions and number of features
    update_functions = []
    if model._node_fn is not None:
        update_functions.append((model._node_fn, n_node_in, n_node_out))
    if model._edge_fn is not None:
        update_functions.append((model._edge_fn, n_edge_in, n_edge_out))
    if model._global_fn is not None:
        update_functions.append((model._global_fn, n_global_in, n_global_out))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over update functions
    for update_fn, n_features_in, n_features_out in update_functions:
        # Loop over update function modules
        for name, module in update_fn.named_children():
            # Check feed-forward neural network
            if name == 'FNN':
                # Get feed-forward neural network layers
                layers = [layer for layer in module.named_children()
                          if 'Layer-' in layer[0]]
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Check number of layers
                if len(layers) != n_hidden_layers + 1:
                    errors.append('Number of layers was not properly set.')
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Loop over layers
                for i, layer in enumerate(layers):
                    # Check input layer features              
                    if i == 0:
                        if layer[1].in_features != n_features_in:
                            errors.append('Number of input features of '
                                          'input layer was not properly set.')
                        if len(layers) == 1:
                            if layer[1].out_features != n_features_out:
                                errors.append('Number of ouput features of '
                                              'input layer was not properly '
                                              'set.')
                        else:
                            if layer[1].out_features != hidden_layer_size:
                                errors.append('Number of ouput features of '
                                              'input layer was not properly '
                                              'set.')
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Check output layer features
                    elif i == len(layers) - 1:
                        if layer[1].in_features != hidden_layer_size:
                            errors.append('Number of input features of output '
                                          'layer was not properly set.')
                        if layer[1].out_features != n_features_out:
                            errors.append('Number of output features of '
                                          'output layer was not properly set.')
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Check hidden layer features
                    else:
                        if layer[1].in_features != hidden_layer_size \
                                or layer[1].out_features != hidden_layer_size:
                            errors.append('Number of input/output features of '
                                          'hidden layer was not properly set.')
            elif name == 'RNN':
                # Get recurrent neural network modules
                layer_name = rnn_cell + '-'
                layers = [layer for layer in module.named_children() if
                          layer_name in layer[0]]
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Check number of modules
                if len(layers) != n_hidden_layers:
                    errors.append('Number of RNN layers was not properly set.')
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Loop over modules
                for i, layer in enumerate(layers):
                    # Check input layer features              
                    if i == 0:
                        if layer[1].input_size != n_features_in:
                            errors.append('Number of input features of input '
                                        'RNN layer was not properly set.')  
                        if len(layers) == 1:
                            if layer[1].hidden_size != hidden_layer_size:
                                errors.append('Number of hidden features of '
                                            'input RNN layer was not properly '
                                            'set.')
                        else:
                            if layer[1].hidden_size != hidden_layer_size:
                                errors.append('Number of hidden features of '
                                            'input RNN layer was not properly '
                                            'set.')
                    else:
                        if layer[1].input_size != hidden_layer_size:
                            errors.append('Number of input features of RNN '
                                        'layer was not properly set.')
                        if layer[1].hidden_size != hidden_layer_size:
                                errors.append('Number of hidden features of '
                                            'RNN layer was not properly set.')
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Check output layer features
                output_layer = [layer for layer in module.named_children() 
                                if 'Output-layer' in layer[0]][0]
                if hidden_layer_size != 0:
                    if output_layer[1].in_features != hidden_layer_size:
                        errors.append('Number of input features of output '
                                        'layer was not properly set.')
                if output_layer[1].out_features != n_features_out:
                    errors.append('Number of output features of output layer '
                                        'was not properly set.')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Check normalization layer
            elif name == 'Norm-Layer':
                if isinstance(module, torch.nn.BatchNorm1d):
                    if module.num_features != n_features_out:
                        errors.append('Number of features of normalization '
                                      'layer was not properly set.')
                elif isinstance(module, torch.nn.LayerNorm):
                    if module.normalized_shape != (n_features_out,):
                        errors.append('Number of features of normalization '
                                      'layer was not properly set.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    assert not errors, "Errors:\n{}".format("\n".join(errors))
# -----------------------------------------------------------------------------
@pytest.mark.parametrize('n_nodes, n_node_in, n_node_out,'
                         'n_edges, n_edge_in, n_edge_out,'
                         'n_global_in, n_global_out,'
                         'n_hidden_layers, hidden_layer_size,'
                         'n_time_node, n_time_edge, n_time_global,'
                         'is_skip_unset_update',
                         [(10, 1, 5, 20, 2, 3, 0, 0, 2, 4, 0, 0, 0, False),
                          (2, 3, 2, 2, 1, 4, 3, 2, 1, 2, 0, 0, 0, False),
                          (3, 2, 4, 6, 5, 4, 0, 0, 0, 2, 0, 0, 0, False),
                          (3, 0, 0, 6, 5, 4, 1, 3, 0, 2, 0, 0, 0, False),
                          (2, 3, 2, 2, 0, 0, 0, 0, 1, 2, 0, 0, 0, False),
                          (3, 0, 0, 6, 5, 4, 2, 3, 0, 2, 0, 0, 0, True),
                          (2, 3, 2, 2, 0, 0, 0, 0, 1, 2, 0, 0, 0, True),
                          (10, 1, 5, 20, 2, 3, 0, 0, 2, 4, 5, 5, 0, False),
                          (2, 3, 2, 2, 1, 4, 3, 2, 1, 2, 2, 2, 2, False),
                          (3, 2, 4, 6, 5, 4, 0, 0, 0, 2, 4, 0, 0, False),
                          (3, 0, 0, 6, 5, 4, 1, 3, 0, 2, 2, 0, 0, False),
                          (2, 3, 2, 2, 0, 0, 0, 0, 1, 2, 5, 0, 5, False),
                          (3, 0, 0, 6, 5, 4, 2, 3, 0, 2, 0, 0, 5, True),
                          (2, 3, 2, 2, 0, 0, 0, 0, 1, 2, 0, 5, 5, True),
                          (3, 0, 0, 6, 5, 4, 1, 3, 0, 2, 3, 3, 0, False),
                          (2, 3, 2, 2, 0, 0, 0, 0, 1, 2, 0, 7, 0, False),
                          ])
def test_graph_independent_network_forward(n_nodes, n_node_in, n_node_out,
                                           n_edges, n_edge_in, n_edge_out,
                                           n_global_in, n_global_out,
                                           n_time_node, n_time_edge,
                                           n_time_global,
                                           n_hidden_layers, hidden_layer_size,
                                           is_skip_unset_update):
    """Test Graph Independent Network forward propagation."""
    # Initialize errors
    errors = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build Graph Independent Network
    model = GraphIndependentNetwork(
        n_hidden_layers=n_hidden_layers, hidden_layer_size=hidden_layer_size,
        n_node_in=n_node_in, n_node_out=n_node_out,
        n_edge_in=n_edge_in, n_edge_out=n_edge_out,
        n_global_in=n_global_in, n_global_out=n_global_out,
        n_time_node=n_time_node, n_time_edge=n_time_edge,
        n_time_global=n_time_global,
        node_hidden_activation=torch.nn.ReLU(),
        node_output_activation=torch.nn.Identity(),
        edge_hidden_activation=torch.nn.ReLU(),
        edge_output_activation=torch.nn.Identity(),
        global_hidden_activation=torch.nn.ReLU(),
        global_output_activation=torch.nn.Identity(),
        is_skip_unset_update=is_skip_unset_update)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate random nodes features input matrix
    node_features_in = None
    if isinstance(n_node_in, int):
        if n_time_node > 0:
            node_features_in = torch.rand(n_nodes, n_node_in*n_time_node)
        else:
            node_features_in = torch.rand(n_nodes, n_node_in)
    # Generate random edges features input matrix
    edge_features_in = None
    if isinstance(n_edge_in, int):
        if n_time_edge > 0:
            edge_features_in = torch.rand(n_edges, n_edge_in*n_time_edge)
        else:
            edge_features_in = torch.rand(n_edges, n_edge_in)
    # Generate random global features input matrix
    global_features_in = None
    if isinstance(n_global_in, int):
        if n_time_global > 0:
            global_features_in = torch.rand(1, n_global_in*n_time_global)
        else:
            global_features_in = torch.rand(1, n_global_in)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Forward propagation
    node_features_out, edge_features_out, global_features_out = model(
        node_features_in=node_features_in, edge_features_in=edge_features_in,
        global_features_in=global_features_in)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check node features output matrix
    if model._node_fn is not None:
        if not isinstance(node_features_out, torch.Tensor):
            errors.append('Nodes features output matrix is not torch.Tensor.')
        elif n_time_node == 0 and not torch.equal(
                        torch.tensor(node_features_out.size()),
                        torch.tensor([n_nodes, n_node_out])):
            errors.append('Nodes features output matrix is not '
                          'torch.Tensor(2d) of shape (n_nodes, n_features).')
        elif n_time_node != 0 and not torch.equal(
                torch.tensor(node_features_out.size()), 
                torch.tensor([n_nodes, n_node_out*n_time_node])):
            errors.append('Nodes features output matrix is not '
                          'torch.Tensor(2d) of shape (n_nodes, '
                          'n_features*n_time_node).')
    else:
        if node_features_in is not None and is_skip_unset_update:
            if not torch.allclose(node_features_out, node_features_in):
                errors.append('Nodes features output matrix is not '
                              'equal to nodes features input matrix.')
        else:
            if node_features_out is not None:
                errors.append('Nodes features output matrix is not None.')
    # Check edge features output matrix
    if model._edge_fn is not None:
        if not isinstance(edge_features_out, torch.Tensor):
            errors.append('Edges features output matrix is not torch.Tensor.')
        elif n_time_edge == 0 and not torch.equal(
                            torch.tensor(edge_features_out.size()),
                            torch.tensor([n_edges, n_edge_out])):
            errors.append('Edges features output matrix is not '
                          'torch.Tensor(2d) of shape (n_edges, n_features).')
        elif n_time_edge != 0 and not torch.equal(
                            torch.tensor(edge_features_out.size()),
                            torch.tensor([n_edges, n_edge_out*n_time_edge])):
            errors.append('Edges features output matrix is not '
                          'torch.Tensor(2d) of shape (n_edges, '
                          'n_features*n_time_edge).')
    else:
        if edge_features_in is not None and is_skip_unset_update:
            if not torch.allclose(edge_features_out, edge_features_in):
                errors.append('Edges features output matrix is not '
                              'equal to edges features input matrix.')
        else:
            if edge_features_out is not None:
                errors.append('Edges features output matrix is not None.')
    # Check global features output matrix
    if model._global_fn is not None:
        if not isinstance(global_features_out, torch.Tensor):
            errors.append('Global features output matrix is not torch.Tensor.')
        elif n_time_global == 0 and not torch.equal(
                            torch.tensor(global_features_out.size()),
                            torch.tensor([1, n_global_out])):
            errors.append('Global features output matrix is not '
                          'torch.Tensor(2d) of shape (1, n_features).')
        elif n_time_global != 0 and not torch.equal(
                            torch.tensor(global_features_out.size()),
                            torch.tensor([1, n_global_out*n_time_global])):
            errors.append('Global features output matrix is not '
                          'torch.Tensor(2d) of shape (1, '
                          'n_features*n_time_global).')
    else:
        if global_features_in is not None and is_skip_unset_update:
            if not torch.allclose(global_features_out, global_features_in):
                errors.append('Global features output matrix is not '
                              'equal to global features input matrix.')
        else:
            if global_features_out is not None:
                errors.append('Global features output matrix is not None.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    assert not errors, "Errors:\n{}".format("\n".join(errors))
# -----------------------------------------------------------------------------
@pytest.mark.parametrize('n_node_in, n_node_out, n_edge_in, n_edge_out,'
                         'n_global_in, n_global_out,'
                         'n_hidden_layers, hidden_layer_size,'
                         'n_time_node, n_time_edge, n_time_global,'
                         'edge_to_node_aggr, node_to_global_aggr, rnn_cell',
                         [(1, 5, 2, 3, 0, 0, 2, 4, 0, 0, 0, 'add', 'add', ''),
                          (3, 2, 1, 4, 0, 2, 1, 2, 0, 0, 0, 'add', 'add', ''),
                          (2, 4, 5, 4, 2, 3, 0, 2, 0, 0, 0, 'add', 'add', ''),
                          (0, 4, 5, 4, 1, 0, 0, 2, 0, 0, 0, 'add', 'add', ''),
                          (2, 4, 0, 4, 0, 0, 0, 2, 0, 0, 0, 'add', 'add', ''),
                          (0, 3, 5, 4, 3, 1, 1, 2, 0, 5, 5,
                            'add', 'add', 'GRU'),
                          (2, 4, 0, 4, 0, 0, 1, 2, 3, 3, 0,
                            'add', 'add', 'GRU'),
                          (1, 5, 2, 3, 0, 0, 2, 4, 5, 5, 0,
                            'add', 'add', 'GRU'),
                          (0, 2, 1, 4, 2, 3, 1, 2, 2, 2, 2,
                            'add', 'add', 'GRU'),
                          (3, 2, 0, 5, 0, 0, 1, 2, 4, 0, 0,
                            'add', 'add', 'GRU'),
                          (1, 1, 2, 3, 5, 5, 3, 7, 0, 0, 5,
                           'add', 'add', 'GRU'),
                          (3, 3, 0, 4, 2, 3, 1, 2, 0, 7, 0,
                            'add', 'add', 'GRU'), 
                          (1, 5, 2, 3, 0, 0, 2, 4, 2, 0, 0,
                            'add', 'add', 'GRU'),
                          (3, 2, 7, 1, 0, 0, 1, 2, 5, 0, 5,
                            'add', 'add', 'GRU'),
                          ])
def test_graph_interaction_network_init(n_node_in, n_node_out, n_edge_in,
                                        n_edge_out, n_global_in, n_global_out,
                                        n_hidden_layers, hidden_layer_size,
                                        n_time_node, n_time_edge, 
                                        n_time_global,
                                        edge_to_node_aggr,
                                        node_to_global_aggr, rnn_cell):
    """Test Graph Interaction Network constructor."""
    # Initialize errors
    errors = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build Graph Interaction Network
    model = GraphInteractionNetwork(
        n_node_out=n_node_out, n_edge_out=n_edge_out,
        n_hidden_layers=n_hidden_layers, hidden_layer_size=hidden_layer_size,
        n_node_in=n_node_in, n_edge_in=n_edge_in, n_global_in=n_global_in,
        n_global_out=n_global_out, edge_to_node_aggr=edge_to_node_aggr,
        n_time_node=n_time_node, n_time_edge=n_time_edge,
        n_time_global=n_time_global,
        node_to_global_aggr=node_to_global_aggr,
        node_hidden_activation=torch.nn.ReLU(),
        node_output_activation=torch.nn.Identity(),
        edge_hidden_activation=torch.nn.ReLU(),
        edge_output_activation=torch.nn.Identity(),
        global_hidden_activation=torch.nn.ReLU(),
        global_output_activation=torch.nn.Identity())
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set Graph Interaction Network update functions and number of features
    update_functions = []
    if model._node_fn is not None:
        update_functions.append((model._node_fn, n_node_in+n_edge_out,
                                 n_node_out))
    if model._edge_fn is not None:
        update_functions.append((model._edge_fn, n_edge_in+2*n_node_in,
                                 n_edge_out))
    if model._global_fn is not None:
        update_functions.append((model._global_fn, n_global_in+n_node_out,
                                 n_global_out))
    # Loop over update functions
    for update_fn, n_features_in, n_features_out in update_functions:        
        # Loop over update function modules
        for name, module in update_fn.named_children():
            # Check feed-forward neural network
            if name == 'FNN':
                # Get feed-forward neural network layers
                layers = [layer for layer in module.named_children()
                          if 'Layer-' in layer[0]]
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Check number of layers
                if len(layers) != n_hidden_layers + 1:
                    errors.append('Number of layers was not properly set.')
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Loop over layers
                for i, layer in enumerate(layers):
                    # Check input layer features              
                    if i == 0:
                        if layer[1].in_features != n_features_in:
                            errors.append('Number of input features of '
                                          'input layer was not properly set.')
                        if len(layers) == 1:
                            if layer[1].out_features != n_features_out:
                                errors.append('Number of ouput features of '
                                              'input layer was not properly '
                                              'set.')
                        else:
                            if layer[1].out_features != hidden_layer_size:
                                errors.append('Number of ouput features of '
                                              'input layer was not properly '
                                              'set.')
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Check output layer features
                    elif i == len(layers) - 1:
                        if layer[1].in_features != hidden_layer_size:
                            errors.append('Number of input features of output '
                                          'layer was not properly set.')
                        if layer[1].out_features != n_features_out:
                            errors.append('Number of output features of '
                                          'output layer was not properly set.')
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Check hidden layer features
                    else:
                        if layer[1].in_features != hidden_layer_size \
                                or layer[1].out_features != hidden_layer_size:
                            errors.append('Number of input/output features of '
                                          'hidden layer was not properly set.')
            elif name == 'RNN':
                # Get recurrent neural network modules
                layer_name = rnn_cell + '-'
                layers = [layer for layer in module.named_children() if 
                          layer_name in layer[0]]
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Check number of modules
                if len(layers) != n_hidden_layers:
                    errors.append('Number of RNN layers was not properly set.')
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Loop over modules
                for i, layer in enumerate(layers):
                    # Check input layer features              
                    if i == 0:
                        if layer[1].input_size != n_features_in:
                            errors.append('Number of input features of input '
                                        'RNN layer was not properly set.')
                        if len(layers) == 1:
                            if layer[1].hidden_size != hidden_layer_size:
                                errors.append('Number of hidden features of '
                                            'input RNN layer was not properly '
                                            'set.')
                        else:
                            if layer[1].hidden_size != hidden_layer_size:
                                errors.append('Number of hidden features of '
                                    'input RNN layer was not properly set.')
                    else:
                        if layer[1].input_size != hidden_layer_size:
                            errors.append('Number of input features of '
                                    'RNN layer was not properly set.')
                        if layer[1].hidden_size != hidden_layer_size:
                                errors.append('Number of hidden features of '
                                            'RNN layer was not properly set.')
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Check output layer features
                output_layer = [layer for layer in module.named_children() 
                                if 'Output-layer' in layer[0]][0]
                if hidden_layer_size != 0:
                    if output_layer[1].in_features != hidden_layer_size:
                        errors.append('Number of input features of RNN output '
                                        'layer was not properly set.')
                if output_layer[1].out_features != n_features_out:
                    errors.append('Number of output features of RNN output '
                                        'layer was not properly set.')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Check normalization layer
            elif name == 'Norm-Layer':
                if isinstance(module, torch.nn.BatchNorm1d):
                    if module.num_features != n_features_out:
                        errors.append('Number of features of normalization '
                                      'layer was not properly set.')
                elif isinstance(module, torch.nn.LayerNorm):
                    if module.normalized_shape != (n_features_out,):
                        errors.append('Number of features of normalization '
                                      'layer was not properly set.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    assert not errors, "Errors:\n{}".format("\n".join(errors))
# -----------------------------------------------------------------------------
@pytest.mark.parametrize('n_nodes, n_node_in, n_node_out,'
                         'n_edges, n_edge_in, n_edge_out,'
                         'n_global_in, n_global_out,'
                         'n_hidden_layers, hidden_layer_size,'
                         'n_time_node, n_time_edge, n_time_global,'
                         'edge_to_node_aggr, node_to_global_aggr',
                         [(10, 1, 5, 20, 2, 3, 0, 0, 2, 4, 0, 0, 0, 
                           'add', 'add'),
                          (4, 3, 2, 2, 1, 4, 0, 2, 1, 2, 0, 0, 0, 
                           'add', 'add'),
                          (3, 2, 4, 6, 5, 4, 1, 0, 0, 2, 0, 0, 0, 
                           'add', 'add'),
                          (4, 0, 1, 2, 1, 1, 2, 3, 1, 2, 0, 0, 0, 
                           'add', 'add'),
                          (3, 2, 4, 6, 0, 4, 0, 0, 0, 2, 0, 0, 0, 
                           'add', 'add'),
                          (4, 3, 2, 2, 1, 4, 7, 2, 1, 2, 4, 0, 0,
                            'add', 'add'),
                          (4, 2, 1, 2, 1, 1, 2, 3, 1, 2, 0, 4, 0,
                            'add', 'mean'),
                          (4, 3, 2, 2, 1, 4, 7, 2, 1, 2, 0, 0, 4,
                            'add', 'add'),
                          (4, 5, 1, 2, 1, 1, 2, 3, 1, 2, 4, 4, 0,
                            'add', 'mean'),
                          (4, 7, 1, 2, 1, 1, 2, 3, 1, 2, 0, 4, 4,
                            'add', 'mean'),
                          (4, 9, 1, 2, 1, 1, 2, 3, 1, 2, 4, 4, 4,
                            'add', 'mean')
                          ])
def test_graph_interaction_network_forward(n_nodes, n_node_in, n_node_out,
                                           n_edges, n_edge_in, n_edge_out,
                                           n_global_in, n_global_out,
                                           n_time_node, n_time_edge,
                                           n_time_global,
                                           n_hidden_layers, hidden_layer_size,
                                           edge_to_node_aggr,
                                           node_to_global_aggr):
    """Test Graph Interaction Network forward propagation."""
    # Initialize errors
    errors = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build Graph Interaction Network
    model = GraphInteractionNetwork(
        n_node_out=n_node_out, n_edge_out=n_edge_out,
        n_hidden_layers=n_hidden_layers, hidden_layer_size=hidden_layer_size,
        n_node_in=n_node_in, n_edge_in=n_edge_in, n_global_in=n_global_in,
        n_global_out=n_global_out, edge_to_node_aggr=edge_to_node_aggr,
        n_time_node=n_time_node, n_time_edge=n_time_edge,
        n_time_global=n_time_global,
        node_to_global_aggr=node_to_global_aggr,
        node_hidden_activation=torch.nn.ReLU(),
        node_output_activation=torch.nn.Identity(),
        edge_hidden_activation=torch.nn.ReLU(),
        edge_output_activation=torch.nn.Identity(),
        global_hidden_activation=torch.nn.ReLU(),
        global_output_activation=torch.nn.Identity())
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set time dimensions
    n_time_in = 0
    if n_time_node > 0: 
        n_time_in = n_time_node
    if n_time_edge > 0:
        n_time_in = n_time_edge
    n_time_all = max(n_time_in, n_time_global)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate random nodes features input matrix
    node_features_in = torch.empty(n_nodes, 0)
    if n_node_in > 0:
        if n_time_node > 0:
            node_features_in = torch.rand(n_nodes, n_node_in*n_time_node)
        else:
            node_features_in = torch.rand(n_nodes, n_node_in)
    # Generate random edges features input matrix
    edge_features_in = None
    if n_edge_in > 0:
        if n_time_edge > 0:
            edge_features_in = torch.rand(n_edges, n_edge_in*n_time_edge)
        else:
            edge_features_in = torch.rand(n_edges, n_edge_in)
    # Generate random edges indexes
    edges_indexes = torch.randint(low=0, high=n_nodes, size=(2, n_edges),
                                  dtype=torch.long)
    # Generate random global features input matrix
    global_features_in = None
    if n_global_in > 0:
        if n_time_global > 0:
            global_features_in = torch.rand(1, n_global_in*n_time_global)
        else:
            global_features_in = torch.rand(1, n_global_in)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Forward propagation
    node_features_out, edge_features_out, global_features_out = \
        model(edges_indexes=edges_indexes,
              node_features_in=node_features_in,
              edge_features_in=edge_features_in,
              global_features_in=global_features_in)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check node features output matrix
    if not isinstance(node_features_out, torch.Tensor):
        errors.append('Nodes features output matrix is not torch.Tensor.')
    elif n_time_in == 0 and not torch.equal(
                        torch.tensor(node_features_out.size()),
                        torch.tensor([n_nodes, n_node_out])):
        errors.append('Nodes features output matrix is not torch.Tensor(2d) '
                      'of shape (n_nodes, n_features).')
    elif n_time_in != 0 and not torch.equal(
                        torch.tensor(node_features_out.size()),
                        torch.tensor([n_nodes, n_node_out*n_time_in])):
        errors.append('Nodes features output matrix is not torch.Tensor(2d) '
                      'of shape (n_nodes, n_features*n_time_node/edge).')
    # Check edge features output matrix
    if not isinstance(edge_features_out, torch.Tensor):
        errors.append('Edges features output matrix is not torch.Tensor.')
    elif n_time_in == 0 and not torch.equal(
                        torch.tensor(edge_features_out.size()),
                        torch.tensor([n_edges, n_edge_out])):
        errors.append('Edges features output matrix is not torch.Tensor(2d) '
                      'of shape (n_edges, n_features).')
    elif n_time_in != 0 and not torch.equal(
                        torch.tensor(edge_features_out.size()),
                        torch.tensor([n_edges, n_edge_out*n_time_in])):
        errors.append('Edges features output matrix is not torch.Tensor(2d) '
                      'of shape (n_edges, n_features*n_time_node/edge).')
    # Check global features output matrix
    if model._global_fn is not None:
        if not isinstance(global_features_out, torch.Tensor):
            errors.append('Global features output matrix is not torch.Tensor.')
        elif n_time_all == 0 and not torch.equal(
                            torch.tensor(global_features_out.size()),
                            torch.tensor([1, n_global_out])):
            errors.append('Global features output matrix is not '
                          'torch.Tensor(2d) of shape (1, n_features).')
        elif n_time_all != 0 and not torch.equal(
                            torch.tensor(global_features_out.size()),
                            torch.tensor([1, n_global_out*n_time_all])):
            errors.append('Global features output matrix is not '
                          'torch.Tensor(2d) of shape (1, '
                          'n_features*n_time_node/edge/global).')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    assert not errors, "Errors:\n{}".format("\n".join(errors))
# -----------------------------------------------------------------------------
@pytest.mark.parametrize('n_nodes, n_node_in, n_node_out,'
                         'n_edges, n_edge_in, n_edge_out,'
                         'n_global_in, n_global_out,'
                         'n_hidden_layers, hidden_layer_size,'
                         'n_time_node, n_time_edge, n_time_global,'
                         'edge_to_node_aggr, node_to_global_aggr',
                         [(10, 1, 5, 20, 2, 3, 0, 0, 2, 4, 0, 0, 0,
                           'add', 'add'),
                          (4, 3, 2, 2, 1, 4, 0, 2, 1, 2, 0, 0, 0,
                            'add', 'add'),
                          (3, 2, 4, 6, 5, 4, 1, 0, 0, 2, 0, 0, 0,
                            'add', 'add'),
                          (4, 0, 1, 2, 1, 1, 2, 3, 1, 2, 0, 0, 0,
                            'add', 'add'),
                          (3, 2, 4, 6, 0, 4, 0, 0, 0, 2, 0, 0, 0,
                            'add', 'add'),
                          (4, 3, 2, 2, 1, 4, 7, 2, 1, 2, 4, 0, 0,
                            'add', 'add'),
                          (4, 2, 1, 2, 1, 1, 2, 3, 1, 2, 0, 4, 0,
                            'add', 'mean'),
                          (4, 3, 2, 2, 1, 4, 7, 2, 1, 2, 0, 0, 4,
                            'add', 'add'),
                          (4, 5, 1, 2, 1, 1, 2, 3, 1, 2, 4, 4, 0,
                            'add', 'mean'),
                          (4, 7, 1, 2, 1, 1, 2, 3, 1, 2, 0, 4, 4,
                            'add', 'mean'),
                          (4, 9, 1, 2, 1, 1, 2, 3, 1, 2, 4, 4, 4,
                            'add', 'mean')
                          ])
def test_graph_interaction_network_message(n_nodes, n_node_in, n_node_out,
                                           n_edges, n_edge_in, n_edge_out,
                                           n_global_in, n_global_out,
                                           n_time_node, n_time_edge,
                                           n_time_global,
                                           n_hidden_layers, hidden_layer_size,
                                           edge_to_node_aggr,
                                           node_to_global_aggr):
    """Test Graph Interaction Network message building (edge update)."""
    # Initialize errors
    errors = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build Graph Interaction Network
    model = GraphInteractionNetwork(
        n_node_out=n_node_out, n_edge_out=n_edge_out,
        n_hidden_layers=n_hidden_layers, hidden_layer_size=hidden_layer_size,
        n_node_in=n_node_in, n_edge_in=n_edge_in, n_global_in=n_global_in,
        n_global_out=n_global_out, edge_to_node_aggr=edge_to_node_aggr,
        n_time_node=n_time_node, n_time_edge=n_time_edge,
        n_time_global=n_time_global,
        node_to_global_aggr=node_to_global_aggr,
        node_hidden_activation=torch.nn.ReLU(),
        node_output_activation=torch.nn.Identity(),
        edge_hidden_activation=torch.nn.ReLU(),
        edge_output_activation=torch.nn.Identity(),
        global_hidden_activation=torch.nn.ReLU(),
        global_output_activation=torch.nn.Identity())    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set time dimensions
    n_time = 0
    if n_time_node > 0: 
        n_time = n_time_node
    if n_time_edge > 0:
        n_time = n_time_edge 
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate random source nodes features input matrix
    node_features_in_i = None
    if n_node_in > 0:
        if n_time_node > 0:
            node_features_in_i = torch.rand(n_edges, n_node_in*n_time_node)
        else:
            node_features_in_i = torch.rand(n_edges, n_node_in)
    # Generate random source nodes features input matrix
    node_features_in_j = None
    if n_node_in > 0:
        if n_time_node > 0:
            node_features_in_j = torch.rand(n_edges, n_node_in*n_time_node)
        else:
            node_features_in_j = torch.rand(n_edges, n_node_in)
    # Generate random edges features input matrix
    edge_features_in = None
    if n_edge_in > 0:
        if n_time_edge > 0:
            edge_features_in = torch.rand(n_edges, n_edge_in*n_time_edge)
        else:
            edge_features_in = torch.rand(n_edges, n_edge_in)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Forward propagation (edge update)
    edge_features_out = model.message(node_features_in_i=node_features_in_i,
                                      node_features_in_j=node_features_in_j,
                                      edge_features_in=edge_features_in)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check edge features output matrix
    if not isinstance(edge_features_out, torch.Tensor):
        errors.append('Edges features output matrix is not torch.Tensor.')
    elif n_time == 0 and not torch.equal(
                        torch.tensor(edge_features_out.size()),
                        torch.tensor([n_edges, n_edge_out])):
        errors.append('Edges features output matrix is not torch.Tensor(2d) '
                      'of shape (n_edges, n_features).')
    elif n_time != 0 and not torch.equal(
                        torch.tensor(edge_features_out.size()),
                        torch.tensor([n_edges, n_edge_out*n_time])):
        errors.append('Edges features output matrix is not torch.Tensor(2d) '
                      'of shape (n_edges, n_features*n_time).')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    assert not errors, "Errors:\n{}".format("\n".join(errors))
# -----------------------------------------------------------------------------
@pytest.mark.parametrize('n_nodes, n_node_in, n_node_out,'
                         'n_edges, n_edge_in, n_edge_out,'
                         'n_global_in, n_global_out,'
                         'n_hidden_layers, hidden_layer_size,'
                         'n_time_node, n_time_edge, n_time_global,'
                         'edge_to_node_aggr, node_to_global_aggr',
                         [(10, 1, 5, 20, 2, 3, 0, 0, 2, 4, 0, 0, 0,
                            'add', 'add'),
                          (4, 3, 2, 2, 1, 4, 0, 2, 1, 2, 0, 0, 0,
                            'add', 'add'),
                          (3, 2, 4, 6, 5, 4, 1, 0, 0, 2, 0, 0, 0,
                            'add', 'add'),
                          (4, 0, 1, 2, 1, 1, 2, 3, 1, 2, 0, 0, 0,
                            'add', 'add'),
                          (3, 2, 4, 6, 0, 4, 0, 0, 0, 2, 0, 0, 0,
                            'add', 'add'),
                          (4, 3, 2, 2, 1, 4, 7, 2, 1, 2, 4, 0, 0,
                            'add', 'add'),
                          (4, 1, 3, 2, 1, 1, 2, 3, 1, 2, 0, 4, 0,
                            'add', 'mean'),
                          (4, 3, 2, 2, 1, 4, 7, 2, 1, 2, 0, 0, 4,
                            'add', 'add'),
                          (4, 5, 1, 2, 1, 1, 2, 3, 1, 2, 4, 4, 0,
                            'add', 'mean'),
                          (4, 7, 1, 2, 1, 1, 2, 3, 1, 2, 0, 4, 4,
                            'add', 'mean'),
                          (4, 9, 1, 2, 1, 1, 2, 3, 1, 2, 4, 4, 4,
                            'add', 'mean'),
                          ])
def test_graph_interaction_network_update(n_nodes, n_node_in, n_node_out,
                                          n_edges, n_edge_in, n_edge_out,
                                          n_global_in, n_global_out,
                                          n_time_node, n_time_edge,
                                          n_time_global,
                                          n_hidden_layers, hidden_layer_size,
                                          edge_to_node_aggr,
                                          node_to_global_aggr):
    """Test Graph Interaction Network node features update."""
    # Initialize errors
    errors = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build Graph Interaction Network
    model = GraphInteractionNetwork(
        n_node_out=n_node_out, n_edge_out=n_edge_out,
        n_hidden_layers=n_hidden_layers, hidden_layer_size=hidden_layer_size,
        n_node_in=n_node_in, n_edge_in=n_edge_in, n_global_in=n_global_in,
        n_global_out=n_global_out, edge_to_node_aggr=edge_to_node_aggr,
        n_time_node=n_time_node, n_time_edge=n_time_edge,
        n_time_global=n_time_global,
        node_to_global_aggr=node_to_global_aggr,
        node_hidden_activation=torch.nn.ReLU(),
        node_output_activation=torch.nn.Identity(),
        edge_hidden_activation=torch.nn.ReLU(),
        edge_output_activation=torch.nn.Identity(),
        global_hidden_activation=torch.nn.ReLU(),
        global_output_activation=torch.nn.Identity())
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set time dimensions
    n_time = 0
    if n_time_node > 0: 
        n_time = n_time_node
    if n_time_edge > 0:
        n_time = n_time_edge 
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate random nodes features input matrix (resulting from aggregation)
    if n_time > 0:
        node_features_in_aggr = torch.rand(n_nodes, n_edge_out*n_time)
    else:
        node_features_in_aggr = torch.rand(n_nodes, n_edge_out)
    # Generate random nodes features input matrix
    node_features_in = None
    if n_node_in > 0:
        if n_time_node > 0:
            node_features_in = torch.rand(n_nodes, n_node_in*n_time_node)
        else:
            node_features_in = torch.rand(n_nodes, n_node_in)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Forward propagation (node update)
    node_features_out = \
        model.update(node_features_in_aggr=node_features_in_aggr,
                     node_features_in=node_features_in)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check node features output matrix
    if not isinstance(node_features_out, torch.Tensor):
        errors.append('Nodes features output matrix is not torch.Tensor.')
    elif n_time == 0 and not torch.equal(
                        torch.tensor(node_features_out.size()),
                        torch.tensor([n_nodes, n_node_out])):
        errors.append('Nodes features output matrix is not torch.Tensor(2d) '
                      'of shape (n_nodes, n_features).')
    elif n_time != 0 and not torch.equal(
                        torch.tensor(node_features_out.size()),
                        torch.tensor([n_nodes, n_node_out*n_time])):
        errors.append('Nodes features output matrix is not torch.Tensor(2d) '
                      'of shape (n_nodes, n_features*n_time).')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    assert not errors, "Errors:\n{}".format("\n".join(errors))
# -----------------------------------------------------------------------------
@pytest.mark.parametrize('n_nodes, n_node_in, n_node_out,'
                         'n_edges, n_edge_in, n_edge_out,'
                         'n_global_in, n_global_out,'
                         'n_hidden_layers, hidden_layer_size,'
                         'n_time_node, n_time_edge, n_time_global,'
                         'edge_to_node_aggr, node_to_global_aggr',
                         [(4, 3, 2, 2, 1, 4, 0, 2, 1, 2, 0, 0, 0,
                            'add', 'add'),
                          (4, 0, 1, 2, 1, 1, 2, 3, 1, 2, 0, 0, 0,
                            'add', 'mean'),
                         (4, 3, 2, 2, 1, 4, 7, 2, 1, 2, 4, 0, 0,
                            'add', 'add'),
                         (4, 1, 3, 2, 1, 1, 2, 3, 1, 2, 0, 4, 0,
                            'add', 'mean'),
                         (4, 3, 2, 2, 1, 4, 7, 2, 1, 2, 0, 0, 4,
                            'add', 'add'),
                         (4, 5, 1, 2, 1, 1, 2, 3, 1, 2, 4, 4, 0,
                            'add', 'mean'),
                         (4, 7, 1, 2, 1, 1, 2, 3, 1, 2, 0, 4, 4,
                            'add', 'mean'),
                         (4, 9, 1, 2, 1, 1, 2, 3, 1, 2, 4, 4, 4,
                            'add', 'mean'),
                          ])
def test_graph_interaction_network_update_global(
    n_nodes, n_node_in, n_node_out, n_edges, n_edge_in, n_edge_out,
    n_global_in, n_time_node, n_time_edge, n_time_global,
    n_global_out, n_hidden_layers, hidden_layer_size,
    edge_to_node_aggr, node_to_global_aggr):
    """Test Graph Interaction Network global features update."""
    # Initialize errors
    errors = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build Graph Interaction Network
    model = GraphInteractionNetwork(
        n_node_out=n_node_out, n_edge_out=n_edge_out,
        n_hidden_layers=n_hidden_layers, hidden_layer_size=hidden_layer_size,
        n_node_in=n_node_in, n_edge_in=n_edge_in,
        n_global_in=n_global_in, n_global_out=n_global_out,
        n_time_node=n_time_node, n_time_edge=n_time_edge,
        n_time_global=n_time_global,
        edge_to_node_aggr=edge_to_node_aggr,
        node_to_global_aggr=node_to_global_aggr,
        node_hidden_activation=torch.nn.ReLU(),
        node_output_activation=torch.nn.Identity(),
        edge_hidden_activation=torch.nn.ReLU(),
        edge_output_activation=torch.nn.Identity(),
        global_hidden_activation=torch.nn.ReLU(),
        global_output_activation=torch.nn.Identity())
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set time dimensions
    n_time = 0
    if n_time_node > 0: 
        n_time = n_time_node
    if n_time_edge > 0:
        n_time = n_time_edge
    if n_time_global > 0:
        n_time = n_time_global 
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate random nodes features output matrix
    if n_time_node > 0 or n_time_edge > 0:
        n_time_in = max([n_time_node, n_time_edge])
        node_features_out = torch.rand(n_nodes, n_node_out*n_time_in)
    else:
        node_features_out = torch.rand(n_nodes, n_node_out)
    # Generate random global features input matrix
    global_features_in = None
    if n_global_in > 0:
        if n_time_global > 0:
            global_features_in = torch.rand(1, n_global_in*n_time_global)
        else:
            global_features_in = torch.rand(1, n_global_in)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Forward propagation (global update)
    global_features_out = \
        model.update_global(node_features_out=node_features_out,
                            global_features_in=global_features_in)
    print('node_features_out: ', node_features_out)
    print('global_features_in: ', global_features_in)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check global features output matrix
    if not isinstance(global_features_out, torch.Tensor):
        errors.append('Global features output matrix is not torch.Tensor.')
    elif n_time == 0 and not torch.equal(
                        torch.tensor(global_features_out.size()),
                        torch.tensor([1, n_global_out])):
        errors.append('Global features output matrix is not torch.Tensor(2d) '
                      'of shape (1, n_features).')
    elif n_time != 0 and not torch.equal(
                        torch.tensor(global_features_out.size()),
                        torch.tensor([1, n_global_out*n_time])):
        errors.append('Global features output matrix is not torch.Tensor(2d) '
                      'of shape (1, n_features*n_time).')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    assert not errors, "Errors:\n{}".format("\n".join(errors))