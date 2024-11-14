"""User script: Generate GNN-based data sets."""
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
import shutil
import random
# Local
from gnn_base_model.data.graph_dataset import GNNGraphDataset
from projects.shell_knock_down.gnn_model_tools.gen_graphs_files import \
    generate_dataset_samples_files
from ioput.iostandard import make_directory
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Planning'
# =============================================================================
#
# =============================================================================
def generate_datasets(dataset_csv_file_path, dataset_split_sizes,
                      datasets_dirs, temp_dataset_directory, is_verbose=False):
    """Generate data sets.
    
    Parameters
    ----------
    dataset_csv_file_path : str
        Data set csv file path.
    dataset_split_sizes : dict
        Size (item, float) of each data set type (key, str), where size is a
        fraction contained between 0 and 1. The sum of all sizes cannot be
        greater than 1.
    datasets_dirs : dict
        Directory (item, str) where each data set type (key, str) is stored.
        All existent files are overridden when saving sample data files.
    is_verbose : bool, default=False
        If True, enable verbose output.
    """
    # Set default files and directories storage options
    sample_file_basename, is_save_sample_plot = set_default_saving_options()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Create temporary data set directory (overwrite)
    make_directory(temp_dataset_directory, is_overwrite=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate data set samples files
    temp_dataset_directory, dataset_samples_files = \
        generate_dataset_samples_files(
            temp_dataset_directory, dataset_csv_file_path,
            sample_file_basename=sample_file_basename,
            is_save_sample_plot=is_save_sample_plot, is_verbose=is_verbose)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Split data set samples files into subsets randomly
    dataset_samples_files_subsets = split_samples_files_randomly(
        dataset_samples_files, dataset_split_sizes)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over data set types
    for dataset_type in dataset_split_sizes.keys():
        # Get subset directory
        subset_directory = datasets_dirs[dataset_type]
        # Get subset samples files
        temp_samples_files = dataset_samples_files_subsets[dataset_type]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize subset samples files
        subset_samples_files = []
        # Loop over subset samples files
        for temp_file_path in temp_samples_files:
            # Build subset sample file path
            sample_file_path = os.path.join(subset_directory,
                                            os.path.basename(temp_file_path))
            # Copy subset sample file to subset directory
            shutil.copy(temp_file_path, sample_file_path)
            # Store subset sample file
            subset_samples_files.append(sample_file_path)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Generate GNN-based data set
        dataset = GNNGraphDataset(subset_directory, subset_samples_files,
                                  dataset_basename='graph_dataset',
                                  is_store_dataset=False)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Save GNN-based data set to file
        _ = dataset.save_dataset(is_append_n_sample=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Remove temporary data set directory
    shutil.rmtree(temp_dataset_directory)
# =============================================================================
def split_samples_files_randomly(dataset_samples_files, dataset_split_sizes):
    """Split data set samples files into subsets randomly.
    
    Parameters
    ----------
    dataset_samples_files : list[str]
        Data set samples files paths.
    dataset_split_sizes : dict
        Size (item, float) of each data set type (key, str), where size is a
        fraction contained between 0 and 1. The sum of all sizes cannot be
        greater than 1.
    
    Returns
    -------
    dataset_samples_files_subsets : dict
        Data set samples files (item, list[str]) assigned to each data set type
        (key, str) subset.
    """
    # Check split sizes
    if sum(dataset_split_sizes.values()) > 1.0:
        raise RuntimeError('The sum of the data set split sizes cannot be '
                           'greater than 1.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get total number of data set samples files
    n_total_sample = len(dataset_samples_files)
    # Randomly shuffle data set sample files
    random.shuffle(dataset_samples_files)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize data set samples files subsets
    dataset_samples_files_subsets = {}
    # Initialize sample index
    sample_index = 0
    # Loop over data set types
    for dataset_type, subset_size in dataset_split_sizes.items():
        # Get data set type number of samples
        n_sample = int(subset_size*n_total_sample)
        # Set extraction indexes
        ini_index = sample_index
        end_index = min(sample_index + n_sample, n_total_sample)
        # Extract subset data set samples
        subset_samples_files = dataset_samples_files[ini_index:end_index]
        # Store subset data set samples
        dataset_samples_files_subsets[dataset_type] = subset_samples_files
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update sample index
        sample_index += n_sample
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return dataset_samples_files_subsets
# =============================================================================
def set_default_saving_options():
    """Set default files and directories storage options.
    
    Returns
    -------
    sample_file_basename : str
        Basename of data set sample file. The basename is appended with sample
        index.
    is_save_sample_plot : bool
        Save plot of each sample graph in the same directory where the data set
        is stored.
    """
    sample_file_basename = 'shell_graph'
    is_save_sample_plot = False
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return sample_file_basename, is_save_sample_plot
# =============================================================================
if __name__ == "__main__":
    # Set data set split sizes
    dataset_split_sizes = \
        {'training': 0.7, 'validation': 0.2, 'in_distribution': 0.1}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set data set file path (shells .csv file)
    dataset_csv_file_path = \
        ('/Users/rbarreira/Desktop/Continual_Learning/graphorge/src/'
         'graphorge/projects/shell_knock_down/0_datasets_files/shells.csv')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set case studies base directory
    base_dir = ('/Users/rbarreira/Desktop/Continual_Learning/graphorge/src/'
         'graphorge/projects/shell_knock_down/')
    # Set case study directory
    case_study_name = '1_graph_from_defects'
    case_study_dir = os.path.join(os.path.normpath(base_dir),
                                  f'{case_study_name}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check case study directory
    if not os.path.isdir(case_study_dir):
        raise RuntimeError('The case study directory has not been found:\n\n'
                           + case_study_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize data sets directories
    datasets_dirs = {}
    # Loop over data set types
    for dataset_type in dataset_split_sizes.keys():
        # Set data set directory
        if dataset_type == 'training':
            # Set testing data set directory (training data set)
            dataset_directory = os.path.join(os.path.normpath(case_study_dir),
                                             '1_training_dataset')
        elif dataset_type == 'validation':
            # Set testing data set directory (validation data set)
            dataset_directory = os.path.join(os.path.normpath(case_study_dir),
                                             '2_validation_dataset')
        elif dataset_type == 'in_distribution':
            # Set testing data set directory (in-distribution testing data set)
            dataset_directory = os.path.join(os.path.normpath(case_study_dir),
                                             '5_testing_id_dataset')
        elif dataset_type == 'out_distribution':
            # Set testing data set directory (out-of-distribution testing data
            # set)
            dataset_directory = os.path.join(os.path.normpath(case_study_dir),
                                             '6_testing_od_dataset')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Create data set directory
        make_directory(dataset_directory, is_overwrite=True)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Store data set directory
        datasets_dirs[dataset_type] = dataset_directory
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set temporary data set directory
    temp_dataset_directory = os.path.join(os.path.normpath(case_study_dir),
                                          '0_dataset')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate data sets
    generate_datasets(dataset_csv_file_path, dataset_split_sizes,
                      datasets_dirs, temp_dataset_directory, is_verbose=True)


