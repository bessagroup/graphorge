"""Mesh visualization tools.

Functions
---------
plot_mesh : Plot mesh data.
plot_truth_vs_pred : Plot mesh ground truth and prediction data.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import os
import shutil
# Third-party
import meshio
import numpy as np
import pyvista as pv
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Guillaume Broggi (g.broggi@tudelft.nl)'
__credits__ = ['Guillaume Broggi', ]
__status__ = 'Planning'
# =============================================================================

def plot_mesh(mesh, data=None, component=None, data_name=None, data_unit=None,
              data_support='point', displacement=None, plotter=None, row=0,
              col=0, show_edges=False, clim=None, cpos=None, cmap='viridis',
              plotter_kwargs=None,show_scalar_bar=True, scalar_bar_kwargs=None,
              text_kwargs=None, title_kwargs=None):
    """Plot mesh data.

    Parameters
    ----------
    mesh : pyvista.DataSet
        Mesh object.
    data : np.ndarray(1d, 2d, 3d), default=None
        Data assigned to the mesh points or cells. The data order must match
        the mesh order. If the data is more than 1d, a vector representation
        is assumed (for instance, the strain principal components).
    component : int, default=None
        Specify the component of the data to plot when the data is 2d or 3d.
        If None, the norm of the data is plotted.
    
    Returns
    -------
    pyvista.plotting.actor.Actor
        Actor of the mesh.
    vtk.vtkScalarBarActor
        Scalar bar actor.
    vtk.vtkTextActor
        Text actor.
    vtk.vtkTextActor
        Title actor.
    """
    # Check the mesh
    if not isinstance(mesh, pv.DataSet):
        raise RuntimeError('The mesh must be a PyVista DataSet object.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Make a copy of the mesh
    mesh = mesh.copy()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check if plotter is provided
    if plotter is None:
        plotter = pv.Plotter()
    elif not isinstance(plotter, pv.Plotter):
        raise RuntimeError('The plotter must be a PyVista Plotter object.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Select the plotter row and column (useful for subplots)
    plotter.subplot(row, col)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Add data to mesh if provided
    if data is not None:
        # Generate a scalar name if not provided
        if data_name is None:
            data_name = 'data'
        # Add data as point or cell data
        if data_support == 'point':
            mesh.point_data[data_name] = data
        elif data_support == 'cell':
            mesh.cell_data[data_name] = data
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Add displacement to mesh if provided
    if displacement is not None:
        # Check displacement is broadcastable to the shape (n_points, 3)
        # Encapsulate the operation in a try-except block to avoid
        # manual check of the shape of the displacement array
        try:
            mesh.points += displacement
        except:
            raise ValueError('Displacement must be broadcastable to'
                             f' the shape ({mesh.n_points}, 3).')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check plotter_kwargs
    if plotter_kwargs is None:
        plotter_kwargs = {}
    elif not isinstance(plotter_kwargs, dict):
        raise RuntimeError('The plotter_kwargs must be a dictionary.')
    # Add mesh to plotter
    plotter.add_mesh(mesh=mesh, scalars=data_name, component=component,
                     preference=data_support, show_edges=show_edges,
                     cmap=cmap, clim=clim, show_scalar_bar=False,
                     **plotter_kwargs)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set camera position if provided
    if cpos is not None:
        plotter.camera_position = cpos
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Show a scalar bar if requested
    if show_scalar_bar:
        # Check scalar_bar_kwargs
        if scalar_bar_kwargs is None:
            scalar_bar_kwargs = {}
        if 'title' not in scalar_bar_kwargs:
            # Prepare a scalar bar title using the data name
            title = f'{data_name}'
            # If the data is 2d or 3d, add the component to the title
            if np.ndim(data) > 1:
                title += f' ({component if component is not None else "norm"})'
            # Add the data unit to the title is any
            if data_unit is not None:
                title += f' [{data_unit}]'
            # Set the title to the scalar bar kwargs
            scalar_bar_kwargs['title'] = title
        if 'n_labels' not in scalar_bar_kwargs:
            scalar_bar_kwargs['n_labels'] = 2
        # Add scalar bar
        scalar_bar = plotter.add_scalar_bar(**scalar_bar_kwargs)
    else:
        scalar_bar = None
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Add text to plotter if requested
    if isinstance(text_kwargs, dict):
        text = plotter.add_text(**text_kwargs)
    else:
        text = None
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Add title to plotter if requested
    if isinstance(title_kwargs, dict):
        title = plotter.add_title(**title_kwargs)
    else:
        title = None
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return actors: plotter, scalar bar, text, and title
    return plotter, scalar_bar, text, title

def plot_truth_vs_pred(mesh, truth_data=None, prediction_data=None, data_unit=None,
                       component=None, data_name=None, data_support='point',
                       displacement=None, plotter=None, show_edges=False,
                       clim=None, cpos=None, cmap='viridis',
                       plotter_kwargs=None, show_scalar_bar=True, 
                       scalar_bar_kwargs=None, text_kwargs=None, 
                       title_kwargs=None):
    
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # If no clim is provided, compute the data range including ground truth
    # and prediction data and set clim to the min and max values
    if clim is None:
        clim = []
        if truth_data is not None:
            if np.ndim(truth_data) > 1:
                # If the truth data is 2d or 3d, select the component
                if component is not None:
                    tmp_data = truth_data[:, component]
                else:
                    tmp_data = np.linalg.norm(truth_data, axis=1)
            else:
                tmp_data = truth_data
            # Min and max values of the truth data
            clim.append(tmp_data.min())
            clim.append(tmp_data.max())
        if prediction_data is not None:
            if np.ndim(prediction_data) > 1:
                # If the truth data is 2d or 3d, select the component
                if component is not None:
                    tmp_data = prediction_data[:, component]
                else:
                    tmp_data = np.linalg.norm(prediction_data, axis=1)
            else:
                tmp_data = prediction_data
            # Min and max values of the prediction data
            clim.append(tmp_data.min())
            clim.append(tmp_data.max())
        if clim:
            # If the list is not empty, set clim to the min and max values
            clim = (min(clim), max(clim))
        else:
            # Using an empty list causes side effects, set clim to None
            clim = None      
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize a plotter with two subplots (left and right)
    plotter = pv.Plotter(shape=(1, 2))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Prepare scalar bar kwargs for the left subplot (ground truth)
    if show_scalar_bar:
        # Check scalar_bar_kwargs
        if scalar_bar_kwargs is None:
            truth_scalar_bar_kwargs = {}
        else:
            truth_scalar_bar_kwargs = scalar_bar_kwargs.copy()
        if 'title' not in truth_scalar_bar_kwargs:
            # Prepare a scalar bar title using the data name
            title = f'Ground truth - {data_name}'
            # If the data is 2d or 3d, add the component to the title
            if np.ndim(truth_data) > 1:
                title += f' ({component if component is not None else "norm"})'
            # Add the data unit to the title is any
            if data_unit is not None:
                title += f' [{data_unit}]'
            # Set the title to the scalar bar kwargs
            truth_scalar_bar_kwargs['title'] = title
        # Center the scalar bar
        truth_scalar_bar_kwargs['position_x'] = 0.2
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot the ground truth data on the left subplot (row=0, col=0)
    plotter, _, _, _ = plot_mesh(mesh, data=truth_data, data_name=data_name, 
                                 data_support=data_support, plotter=plotter,
                                 row=0, col=0, show_edges=show_edges, clim=clim,
                                 cpos=cpos, cmap=cmap, 
                                 plotter_kwargs=plotter_kwargs, 
                                 show_scalar_bar=show_scalar_bar,
                                 scalar_bar_kwargs=truth_scalar_bar_kwargs,
                                 text_kwargs=text_kwargs, 
                                 title_kwargs=title_kwargs)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Prepare scalar bar kwargs for the right subplot (prediction)
    if show_scalar_bar:
        # Check scalar_bar_kwargs
        if scalar_bar_kwargs is None:
            pred_scalar_bar_kwargs = {}
        else:
            pred_scalar_bar_kwargs = scalar_bar_kwargs.copy()
        if 'title' not in pred_scalar_bar_kwargs:
            # Prepare a scalar bar title using the data name
            title = f'Prediction - {data_name}'
            # If the data is 2d or 3d, add the component to the title
            if np.ndim(truth_data) > 1:
                title += f' ({component if component is not None else "norm"})'
            # Add the data unit to the title is any
            if data_unit is not None:
                title += f' [{data_unit}]'
            # Set the title to the scalar bar kwargs
            pred_scalar_bar_kwargs['title'] = title
        # Center the scalar bar
        pred_scalar_bar_kwargs['position_x'] = 0.2
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot the prediction data on the right subplot (row=0, col=1)
    # The mesh is deformed if displacement is provided
    plotter, _, _, _ = plot_mesh(mesh, data=prediction_data,
                                 data_name=data_name, data_support=data_support,
                                 displacement=displacement, plotter=plotter,
                                 row=0, col=1, show_edges=show_edges, clim=clim,
                                 cpos=cpos, cmap=cmap,
                                 plotter_kwargs=plotter_kwargs,
                                 show_scalar_bar=show_scalar_bar,
                                 scalar_bar_kwargs=pred_scalar_bar_kwargs)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return the plotter
    return plotter

def load_inp_mesh(inp_file_path):
    """Load mesh from ABAQUS input file.

    Parameters
    ----------
    inp_file_path : str
        ABAQUS input file path.

    Returns
    -------
    mesh : pyvista.PolyData
        Mesh object.
    """
    # Check the inp file
    if not os.path.isfile(inp_file_path):
        raise RuntimeError('The mesh file has not been found:\n\n'
                           + inp_file_path)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load mesh
    return pv.from_meshio(meshio.read(inp_file_path))

