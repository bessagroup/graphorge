"""Hydra hyperparameter optimization.

Execute (multi-run mode):

    $ python3 hydra_optimization.py -m

Functions
---------
hydra_wrapper
    Wrapper of Hydra hyperparameter optimization main function.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import sys
import pathlib
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Add project root directory to sys.path
root_dir = str(pathlib.Path(__file__).parents[3])
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
# Third-party
import hydra
import torch
import numpy as np
# Local
from gnn_base_model.data.graph_dataset import GNNGraphDataset
from gnn_base_model.train.training import train_model
from gnn_base_model.predict.prediction import predict
from gnn_base_model.optimization.hydra_optimization_template import \
    display_hydra_job_header
from ioput.iostandard import make_directory, write_summary_file
from projects.shell_knock_down.user_scripts.train_model import \
    generate_standard_training_plots
from projects.shell_knock_down.user_scripts.predict import \
    generate_prediction_plots
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
# =============================================================================
#
# =============================================================================
def hydra_wrapper(process, dataset_paths, device_type='cpu'):
    """Wrapper of Hydra hyperparameter optimization main function.
    
    Parameters
    ----------
    process : str
        Hyperparameter optimization process.
    dataset_paths : dict
        Hyperparameter optimization process required data sets (key, str)
        file paths (item, str).
    device_type : {'cpu', 'cuda'}, default='cpu'
        Type of device on which torch.Tensor is allocated.
    """
    # Set Hydra main function
    @hydra.main(version_base=None, config_path='.', config_name='hydra_config')
    def hydra_optimize_gnn_model(cfg):
        """Hydra hyperparameter optimization of GNN-based material patch model.
        
        Parameters
        ----------
        cgf : omegaconf.DictConfig
            Configuration dictionary of YAML based hierarchical configuration
            system.
            
        Returns
        -------
        objective : float
            Objective to minimize.
        """
        # Get Hydra configuration singleton
        hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Display Hydra hyperparameter optimization job header
        sweeper, sweeper_optimizer, job_dir = \
            display_hydra_job_header(hydra_cfg)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set GNN-based model initialization parameters
        model_init_args = {}
        model_init_args['n_node_in'] = cfg.n_node_in
        model_init_args['n_node_out'] = cfg.n_node_out
        model_init_args['n_edge_in'] = cfg.n_edge_in
        model_init_args['n_edge_out'] = cfg.n_edge_out
        model_init_args['n_global_in'] = cfg.n_global_in
        model_init_args['n_global_out'] = cfg.n_global_out
        model_init_args['n_message_steps'] = cfg.n_message_steps
        model_init_args['enc_n_hidden_layers'] = cfg.n_hidden_layers
        model_init_args['pro_n_hidden_layers'] = cfg.n_hidden_layers
        model_init_args['dec_n_hidden_layers'] = cfg.n_hidden_layers
        model_init_args['hidden_layer_size'] = cfg.hidden_layer_size
        model_init_args['model_directory'] = job_dir
        model_init_args['model_name'] = 'graph_neural_networ_model'
        model_init_args['is_model_in_normalized'] = True
        model_init_args['is_model_out_normalized'] = True
        model_init_args['pro_edge_to_node_aggr'] = cfg.pro_edge_to_node_aggr
        model_init_args['pro_node_to_global_aggr'] = \
            cfg.pro_node_to_global_aggr
        model_init_args['enc_node_hidden_activ_type'] = cfg.hidden_activation
        model_init_args['enc_node_output_activ_type'] = cfg.output_activation
        model_init_args['enc_edge_hidden_activ_type'] = cfg.hidden_activation
        model_init_args['enc_edge_output_activ_type'] = cfg.output_activation
        model_init_args['enc_global_hidden_activ_type'] = cfg.hidden_activation
        model_init_args['enc_global_output_activ_type'] = cfg.output_activation
        model_init_args['pro_node_hidden_activ_type'] = cfg.hidden_activation
        model_init_args['pro_node_output_activ_type'] = cfg.output_activation
        model_init_args['pro_edge_hidden_activ_type'] = cfg.hidden_activation
        model_init_args['pro_edge_output_activ_type'] = cfg.output_activation
        model_init_args['pro_global_hidden_activ_type'] = cfg.hidden_activation
        model_init_args['pro_global_output_activ_type'] = cfg.output_activation
        model_init_args['dec_node_hidden_activ_type'] = cfg.hidden_activation
        model_init_args['dec_node_output_activ_type'] = cfg.output_activation
        model_init_args['dec_edge_hidden_activ_type'] = cfg.hidden_activation
        model_init_args['dec_edge_output_activ_type'] = cfg.output_activation
        model_init_args['dec_global_hidden_activ_type'] = cfg.hidden_activation
        model_init_args['dec_global_output_activ_type'] = cfg.output_activation
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Load hyperparameter optimization process data sets
        if process in ('training', 'training-testing'):
            # Get training data set file path
            train_dataset_file_path = dataset_paths['training']
            # Load training data set
            training_dataset = \
                GNNGraphDataset.load_dataset(train_dataset_file_path)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set early stopping
            is_early_stopping = cfg.is_early_stopping
            # Set early stopping parameters
            if is_early_stopping:
                # Get validation data set file path
                val_dataset_file_path = dataset_paths['validation']
                # Load validation data set
                validation_dataset = \
                    GNNGraphDataset.load_dataset(val_dataset_file_path)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Get early stopping parameters
                early_stopping_kwargs = {**cfg.early_stopping_kwargs}
                # Add validation dataset to early stopping parameters
                early_stopping_kwargs['validation_dataset'] = \
                    validation_dataset
        else:
            raise RuntimeError('Unknown hyperparameter optimization process.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Load hyperparameter optimization process testing data set
        if process in ('training-testing',):
            # Get testing data set file path
            test_dataset_file_path = dataset_paths['testing']
            # Load testing data set
            testing_dataset = \
                GNNGraphDataset.load_dataset(test_dataset_file_path)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Training of GNN-based model
        if process in ('training', 'training-testing'):
            # Set model training subdirectory
            training_subdir = os.path.join(os.path.normpath(job_dir), 'model')
            # Create model training subdirectory
            training_subdir = make_directory(training_subdir,
                                             is_overwrite=True)
            # Set model directory
            model_init_args['model_directory'] = training_subdir
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Training
            model, best_training_loss, _ = train_model(
                cfg.n_max_epochs, training_dataset, model_init_args,
                cfg.lr_init, opt_algorithm=cfg.opt_algorithm,
                lr_scheduler_type=cfg.lr_scheduler_type,
                lr_scheduler_kwargs=cfg.lr_scheduler_kwargs,
                loss_nature=cfg.loss_nature,
                loss_type=cfg.loss_type, loss_kwargs=cfg.loss_kwargs,
                batch_size=cfg.batch_size,
                is_sampler_shuffle=cfg.is_sampler_shuffle,
                is_early_stopping=cfg.is_early_stopping,
                early_stopping_kwargs=early_stopping_kwargs,
                device_type=device_type, is_verbose=False)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Generate plots of model training process
            generate_standard_training_plots(training_subdir)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set hyperparameter optimization objective
            if process == 'training':
                objective = best_training_loss
            elif process == 'training-testing':
                # Set model testing subdirectory
                testing_subdir = \
                    os.path.join(os.path.normpath(job_dir), 'testing')
                # Create model testing subdirectory
                testing_subdir = \
                    make_directory(testing_subdir, is_overwrite=True)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Set prediction loss normalization
                is_normalized_loss = False
                # Set prediction batch size
                batch_size = len(testing_dataset)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Testing of GNN-based material patch model
                predict_subdir, avg_valid_loss_sample = predict(
                    testing_dataset, model.model_directory, model=model,
                    predict_directory=testing_subdir,
                    load_model_state='best', loss_nature=cfg.loss_nature,
                    loss_type=cfg.loss_type, loss_kwargs=cfg.loss_kwargs,
                    is_normalized_loss=is_normalized_loss,
                    batch_size=batch_size, device_type=device_type, 
                    is_verbose=False)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Generate plots of model predictions
                generate_prediction_plots(predict_subdir)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Set hyperparameter optimization objective
                objective = avg_valid_loss_sample
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Display parameters
        print('\nParameters:')
        for key, val in cfg.items():
            print(f'  > {key:{max([len(x) for x in cfg.keys()])}} : {val}')
        # Display objective
        print(f'\nFunction evaluation:')
        print(f'  > Objective : {objective:.8e}\n')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set summary data
        summary_data = {}
        summary_data['sweeper'] = sweeper
        summary_data['sweeper_optimizer'] = sweeper_optimizer
        for key, val in cfg.items():
            summary_data[key] = val
        summary_data['objective'] = f'{objective:.8e}'
        # Write summary file
        write_summary_file(\
            summary_directory=job_dir,
            filename='job_summary',
            summary_title='Hydra - Hyperparameter Optimization Job',
            **summary_data)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return objective
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Call Hydra main function
    hydra_optimize_gnn_model()
# =============================================================================
if __name__ == "__main__":
    # Set hyperparameter optimization processes
    processes = ('training', 'training-testing')
    # Select hyperparameter optimization process
    process = processes[1]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set optimization process data set paths
    datasets_paths = {}
    if process == 'training':
        datasets_paths['training'] = None
    elif process == 'training-testing':
        datasets_paths['training'] = \
            ('/home/bernardoferreira/Documents/brown/projects/'
             'colaboration_guillaume/shell_knock_down/case_studies/debug_gen/'
             '1_training_dataset/graph_dataset_n5243.pkl')
        datasets_paths['validation'] = \
            ('/home/bernardoferreira/Documents/brown/projects/'
             'colaboration_guillaume/shell_knock_down/case_studies/debug_gen/'
             '2_validation_dataset/graph_dataset_n1498.pkl')
        datasets_paths['testing'] = \
            ('/home/bernardoferreira/Documents/brown/projects/'
             'colaboration_guillaume/shell_knock_down/case_studies/debug_gen/'
             '5_testing_id_dataset/graph_dataset_n749.pkl')
    else:
        raise RuntimeError('Unknown hyperparameter optimization process.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set device type
    if torch.cuda.is_available():
        device_type = 'cuda'
    else:
        device_type = 'cpu'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Execute Hydra hyperparameter optimization
    hydra_wrapper(process, datasets_paths, device_type)