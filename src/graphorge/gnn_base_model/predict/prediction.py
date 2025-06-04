"""Prediction of Graph Neural Network model.

Functions
---------
predict
    Make predictions with Graph Neural Network model for given dataset.
make_predictions_subdir
    Create model predictions subdirectory.
save_sample_predictions
    Save model prediction results for given sample.
load_sample_predictions
    Load model prediction results for given sample.
compute_sample_prediction_loss
    Compute loss of sample output features prediction.
seed_worker
    Set workers seed in PyTorch data loaders to preserve reproducibility.
write_prediction_summary_file
    Write summary data file for model prediction process.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import os
import pickle
import random
import re
import time
import datetime
# Third-party
import torch
import torch_geometric
from tqdm import tqdm
import numpy as np
# Local
from gnn_base_model.model.gnn_model import GNNEPDBaseModel
from gnn_base_model.train.torch_loss import get_pytorch_loss
from ioput.iostandard import make_directory, write_summary_file
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
# =============================================================================
#
# =============================================================================
def predict(dataset, model_directory, model=None, predict_directory=None,
            file_name_pattern=None, load_model_state=None,
            loss_nature='node_features_out', loss_type='mse', loss_kwargs={},
            is_normalized_loss=False, batch_size=1, dataset_file_path=None,
            device_type='cpu', seed=None, is_verbose=False):
    """Make predictions with Graph Neural Network model for given dataset.
    
    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Graph Neural Network graph data set. Each sample corresponds to a
        torch_geometric.data.Data object describing a homogeneous graph.
    model_directory : str
        Directory where Graph Neural Network model is stored.
    model : GNNEPDBaseModel, default=None
        Graph Neural Network model. If None, then model is initialized
        from the initialization file and the state is loaded from the state
        file. In both cases the model is set to evaluation mode.
    predict_directory : str, default=None
        Directory where model predictions results are stored. If None, then
        all output files are supressed.
    file_name_pattern : str, default=None
        A f-string pattern for the file name used to save prediction results.
    load_model_state : {'best', 'last', int, None}, default=None
        Load available Graph Neural Network model state from the model
        directory. Options:
        
        'best' : Model state corresponding to best performance available
        
        'last' : Model state corresponding to highest training epoch
        
        int    : Model state corresponding to given training epoch
        
        None   : Model default state file

    loss_nature : {'node_features_out', 'global_features_out'}, \
                  default='node_features_out'
        Loss nature:
        
        'node_features_out' : Based on node output features

        'global_features_out' : Based on global output features

    loss_type : {'mse',}, default='mse'
        Loss function type:
        
        'mse'  : MSE (torch.nn.MSELoss)
        
    loss_kwargs : dict, default={}
        Arguments of torch.nn._Loss initializer.
    is_normalized_loss : bool, default=False
        If True, then samples prediction loss are computed from normalized
        output data, False otherwise. Normalization of output data requires
        that model data scalers are available.
    batch_size : int, default=1
        Number of samples loaded per batch.
    dataset_file_path : str, default=None
        Graph Neural Network graph data set file path if such file exists. Only
        used for output purposes.
    device_type : {'cpu', 'cuda'}, default='cpu'
        Type of device on which torch.Tensor is allocated.
    seed : int, default=None
        Seed used to initialize the random number generators of Python and
        other libraries (e.g., NumPy, PyTorch) for all devices to preserve
        reproducibility. Does also set workers seed in PyTorch data loaders.
    is_verbose : bool, default=False
        If True, enable verbose output.
        
    Returns
    -------
    predict_subdir : str
        Subdirectory where samples predictions results files are stored.
    avg_predict_loss : float
        Average prediction loss per sample. Defaults to None if ground-truth is
        not available for all data set samples.
    """
    # Set random number generators initialization for reproducibility
    if isinstance(seed, int):
        random.seed(seed)
        np.random.seed(seed)
        generator = torch.Generator().manual_seed(seed)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set device
    device = torch.device(device_type)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    start_time_sec = time.time()
    if is_verbose:
        print('\nGraph Neural Network model prediction'
              '\n-------------------------------------')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check model directory
    if not os.path.exists(model_directory):
        raise RuntimeError('The Graph Neural Network model directory has not '
                           'been found:\n\n' + model_directory)
    # Check prediction directory
    if predict_directory is not None and not os.path.exists(predict_directory):
        raise RuntimeError('The Graph Neural Network model prediction '
                           'directory has not been found:\n\n'
                           + predict_directory)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize model and load model state if not provided
    if model is None:
        if is_verbose:
            print('\n> Loading Graph Neural Network model...')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize Graph Neural Network model
        model = GNNEPDBaseModel.init_model_from_file(model_directory)
        # Set model device
        model.set_device(device_type)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Load Graph Neural Network model state
        _ = model.load_model_state(load_model_state=load_model_state,
                                   is_remove_posterior=False)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get model input and output features normalization
    is_model_in_normalized = model.is_model_in_normalized
    is_model_out_normalized = model.is_model_out_normalized
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Move model to device
    model.to(device=device)
    # Set model in evaluation mode
    model.eval()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Create model predictions subdirectory for current prediction process
    predict_subdir = None
    if predict_directory is not None:
        predict_subdir = make_predictions_subdir(predict_directory)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set data loader
    if isinstance(seed, int):
        data_loader = torch_geometric.loader.dataloader.DataLoader(
            dataset=dataset, batch_size=batch_size,
            worker_init_fn=seed_worker, generator=generator, shuffle=False)
    else:
        data_loader = torch_geometric.loader.dataloader.DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=False)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize loss function
    loss_function = get_pytorch_loss(loss_type, **loss_kwargs)
    # Initialize samples prediction loss
    loss_samples = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if is_verbose:
        print('\n\n> Starting prediction process...\n')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set context manager to avoid creation of computation graphs during the
    # model evaluation (forward propagation)
    with torch.no_grad():
        # Loop over graph samples
        for i, pyg_graph in enumerate(tqdm(data_loader, mininterval=1,
                                           maxinterval=60, miniters=0, 
                                           desc='> Predictions: ', 
                                           disable=not is_verbose,
                                           unit=' sample')):
            # Move sample to device
            pyg_graph.to(device)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get batch node assignment vector
            if batch_size > 1:
                batch_vector = pyg_graph.batch
            else:
                batch_vector = None
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get input features from input graph
            node_features_in, edge_features_in, global_features_in, \
                edges_indexes = model.get_input_features_from_graph(
                    pyg_graph, is_normalized=is_model_in_normalized)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get node output features ground-truth
            node_targets, edge_targets, global_targets = \
                model.get_output_features_from_graph(
                    pyg_graph, is_normalized=False)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize sample results
            results = {}
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get metadata
            metadata = model.get_metadata_from_graph(pyg_graph)
            # Store metadata
            results['metadata'] = {}
            if isinstance(metadata, dict):
                # Iterate over metadata items
                for key, value in metadata.items():
                    # Process tensor metadata
                    if isinstance(value, torch.Tensor):
                        # If there is only one element, store it as a scalar
                        if value.numel() == 1:
                            results['metadata'][key] = (
                                value.detach().cpu().item())
                        # Otherwise, store it as a numpy array
                        else:
                            results['metadata'][key] = (
                                value.detach().cpu().numpy())
                    # Process non-tensor metadata
                    else:
                        results['metadata'][key] = value
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute output features predictions (forward propagation)
            if loss_nature == 'node_features_out':
                # Compute node output features
                features_out, _, _ = model(
                    node_features_in=node_features_in,
                    edge_features_in=edge_features_in,
                    global_features_in=global_features_in,
                    edges_indexes=edges_indexes, batch_vector=batch_vector)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Denormalize node output features
                if is_model_out_normalized:
                    # Get model data scaler
                    features_out = model.data_scaler_transform(
                        tensor=features_out, features_type='node_features_out',
                        mode='denormalize')
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Get sample node output features ground-truth
                # (None if not available)
                targets = node_targets
                # Store sample results
                results['node_features_out'] = features_out.detach().cpu()
                results['node_targets'] = None
                if targets is not None:
                    results['node_targets'] = targets.detach().cpu()
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            elif loss_nature == 'global_features_out':
                # Compute global output features
                _, _, features_out = model(
                    node_features_in=node_features_in,
                    edge_features_in=edge_features_in,
                    global_features_in=global_features_in,
                    edges_indexes=edges_indexes, batch_vector=batch_vector)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Denormalize global output features
                if is_model_out_normalized:
                    # Get model data scaler
                    features_out = model.data_scaler_transform(
                        tensor=features_out,
                        features_type='global_features_out',
                        mode='denormalize')
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Get sample global output features ground-truth
                # (None if not available)
                targets = global_targets
                # Store sample results
                results['global_features_out'] = features_out.detach().cpu()
                results['global_targets'] = None
                if targets is not None:
                    results['global_targets'] = targets.detach().cpu()
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            else:
                raise RuntimeError('Unknown loss nature.')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute sample output features prediction loss
            loss = compute_sample_prediction_loss(
                model, loss_nature, loss_function, features_out, targets,
                is_normalized_loss=is_normalized_loss)
            # Store prediction loss data
            results['prediction_loss_data'] = \
                (loss_nature, loss_type, loss, is_normalized_loss)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Assemble sample prediction loss if ground-truth is available
            if loss is not None:
                loss_samples.append(loss.detach().cpu())
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Save sample predictions results
            if predict_directory is not None:
                save_sample_predictions(predictions_dir=predict_subdir,
                                        prediction_id=i,
                                        sample_results=results,
                                        file_name_pattern=file_name_pattern)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if is_verbose:
        print('\n> Finished prediction process!\n')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute average prediction loss per sample
    avg_predict_loss = None
    if isinstance(loss_samples, list):
        avg_predict_loss = np.mean(loss_samples)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if is_verbose:
        # Set average prediction loss output format
        if avg_predict_loss:
            loss_str = (f'{avg_predict_loss:.8e} | {loss_type}')
            if is_normalized_loss:
                loss_str += ', normalized'
        else:
            loss_str = 'Ground-truth not available'  
        # Display average loss
        print('\n> Avg. prediction loss per sample: '
              + loss_str)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute total prediction time and average prediction time per sample
    total_time_sec = time.time() - start_time_sec
    if len(dataset) > 0:
        avg_time_sample = total_time_sec/len(dataset)
    else:
        avg_time_sample = float('nan')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if is_verbose:
        print(f'\n> Prediction results directory: {predict_subdir}')
        print(f'\n> Total prediction time: '
              f'{str(datetime.timedelta(seconds=int(total_time_sec)))} | '
              f'Avg. prediction time per sample: '
              f'{str(datetime.timedelta(seconds=int(avg_time_sample)))}\n')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Write summary data file for model prediction process
    if predict_directory is not None:
        write_prediction_summary_file(
            predict_subdir, device_type, seed, model_directory,
            load_model_state, loss_type, loss_kwargs, is_normalized_loss,
            dataset_file_path, dataset, avg_predict_loss, total_time_sec,
            avg_time_sample)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return predict_subdir, avg_predict_loss
# =============================================================================
def make_predictions_subdir(predict_directory):
    """Create model predictions subdirectory.
    
    Parameters
    ----------
    predict_directory : str
        Directory where model predictions results are stored.

    Returns
    -------
    predict_subdir : str
        Subdirectory where samples predictions results files are stored.
    """
    # Check prediction directory
    if not os.path.exists(predict_directory):
        raise RuntimeError('The model prediction directory has not been '
                           'found:\n\n' + predict_directory)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set predictions subdirectory path
    predict_subdir = os.path.join(predict_directory, 'prediction_set_0')
    while os.path.exists(predict_subdir):
        predict_subdir = os.path.join(
            predict_directory,
            'prediction_set_' + str(int(predict_subdir.split('_')[-1]) + 1))
    # Create model predictions subdirectory
    predict_subdir = make_directory(predict_subdir, is_overwrite=False)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return predict_subdir
# =============================================================================
def save_sample_predictions(predictions_dir, prediction_id, sample_results,
                            file_name_pattern = None):
    """Save model prediction results for given sample.
    
    Parameters
    ----------
    predictions_dir : str
        Directory where sample prediction results are stored.
    prediction_id : int
        Prediction ID appended to the prediction sample results file name.
    sample_results : dict
        Sample prediction results.
    file_name_pattern: str, default=None
        A f-string pattern for the file name. The pattern will be evaluated
        when saving the predictions and has access to `prediction_id` and
        all the `sample_results['metadata']` content. If None, the pattern
        ``'prediction_sample_{prediction_id}'`` is used.
    """
    # Check prediction results directory
    if not os.path.exists(predictions_dir):
        raise RuntimeError('The prediction results directory has not been '
                           'found:\n\n' + predictions_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate file name
    if file_name_pattern is None:
        file_name_pattern = 'prediction_sample_{prediction_id}'
    file_name = file_name_pattern.format(**sample_results['metadata'],
                                         prediction_id=prediction_id)
    # Set sample prediction results file path
    sample_path = os.path.join(predictions_dir, file_name + '.pkl')
    # Save sample prediction results
    with open(sample_path, 'wb') as sample_file:
        pickle.dump(sample_results, sample_file)
# =============================================================================
def load_sample_predictions(sample_prediction_path):
    """Load model prediction results for given sample.
    
    Parameters
    ----------
    sample_prediction_path : str
        Sample prediction results file path.
        
    Returns
    -------
    sample_results : dict
        Sample prediction results.
    """
    # Check sample prediction results file
    if not os.path.isfile(sample_prediction_path):
        raise RuntimeError('Sample prediction results file has not been '
                           'found:\n\n' + sample_prediction_path)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load sample prediction results
    with open(sample_prediction_path, 'rb') as sample_prediction_file:
        sample_results = pickle.load(sample_prediction_file)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return sample_results
# =============================================================================
def compute_sample_prediction_loss(model, loss_nature, loss_function,
                                   features_out, targets,
                                   is_normalized_loss=False):
    """Compute loss of sample output features prediction.
    
    Assumes that provided output features and targets are denormalized.
    
    Parameters
    ----------
    model : GNNEPDBaseModel
        Graph Neural Network model.
    loss_nature : {'node_features_out', 'global_features_out'}
        Loss nature.
    loss_function : torch.nn._Loss
        PyTorch loss function.
    features_out : torch.Tensor
        Predicted output features stored as a torch.Tensor(2d).
    targets : {torch.Tensor, None}
        Output features ground-truth stored as a torch.Tensor(2d).
    is_normalized_loss : bool, default=False
        If True, then samples prediction loss are computed from normalized
        output data, False otherwise. Normalization of output data requires
        that model data scalers are available.
    
    Returns
    -------
    loss : {float, None}
        Loss of sample output features prediction. Set to None if output
        features ground-truth is not available.
    """
    # Check if output features ground-truth is available
    is_ground_truth_available = targets is not None
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute sample loss
    if is_ground_truth_available:
        # Normalize output features
        if is_normalized_loss:
            # Get model data scaler
            if loss_nature == 'node_features_out':
                scaler = model.get_fitted_data_scaler('node_features_out')
            elif loss_nature == 'global_features_out':
                scaler = model.get_fitted_data_scaler('global_features_out')
            else:
                raise RuntimeError('Unknown loss nature.')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get normalized output features predictions
            features_out = scaler.transform(features_out)
            # Get normalized output features ground-truth
            targets = scaler.transform(targets)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute sample loss
        loss = loss_function(features_out, targets)
    else:
        # Set sample loss to None if ground-truth is not available
        loss = None
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return loss
# =============================================================================
def seed_worker(worker_id):
    """Set workers seed in PyTorch data loaders to preserve reproducibility.
    
    Taken from: https://pytorch.org/docs/stable/notes/randomness.html
    
    Parameters
    ----------
    worker_id : int
        Worker ID.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
# =============================================================================
def write_prediction_summary_file(
    predict_subdir, device_type, seed, model_directory, load_model_state,
    loss_type, loss_kwargs, is_normalized_loss, dataset_file_path, dataset,
    avg_predict_loss, total_time_sec, avg_time_sample):
    """Write summary data file for model prediction process.
    
    Parameters
    ----------
    predict_subdir : str
        Subdirectory where samples predictions results files are stored.
    device_type : {'cpu', 'cuda'}
        Type of device on which torch.Tensor is allocated.
    seed : int
        Seed used to initialize the random number generators of Python and
        other libraries (e.g., NumPy, PyTorch) for all devices to preserve
        reproducibility. Does also set workers seed in PyTorch data loaders.
    model_directory : str
        Directory where model is stored.
    load_model_state : {'best', 'last', int, None}
        Load availabl model state from the model directory. Data scalers are
        also loaded from model initialization file.
    loss_type : {'mse',}
        Loss function type.
    loss_kwargs : dict
        Arguments of torch.nn._Loss initializer.
    is_normalized_loss : bool, default=False
        If True, then samples prediction loss are computed from the normalized
        data, False otherwise. Normalization requires that model features data
        scalers are fitted.
    dataset_file_path : str
        Data set file path if such file exists. Only used for output purposes.
    dataset : torch.utils.data.Dataset
        Data set.
    avg_predict_loss : float
        Average prediction loss per sample.
    total_time_sec : int
        Total prediction time in seconds.
    avg_time_sample : float
        Average prediction time per sample.
    """
    # Set summary data
    summary_data = {}
    summary_data['device_type'] = device_type
    summary_data['seed'] = seed
    summary_data['model_directory'] = model_directory
    summary_data['load_model_state'] = load_model_state
    summary_data['loss_type'] = loss_type
    summary_data['loss_kwargs'] = loss_kwargs if loss_kwargs else None
    summary_data['is_normalized_loss'] = is_normalized_loss
    summary_data['Prediction data set file'] = \
        dataset_file_path if dataset_file_path else None
    summary_data['Prediction data set size'] = len(dataset)
    summary_data['Avg. prediction loss per sample'] = \
        f'{avg_predict_loss:.8e}' if avg_predict_loss else None
    summary_data['Total prediction time'] = \
        str(datetime.timedelta(seconds=int(total_time_sec)))
    summary_data['Avg. prediction time per sample'] = \
        str(datetime.timedelta(seconds=int(avg_time_sample)))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Write summary file
    write_summary_file(
        summary_directory=predict_subdir,
        summary_title='Summary: Model prediction',
        **summary_data)