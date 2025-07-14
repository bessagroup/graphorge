# Installation

Graphorge is available as a **Python package** which depends on Python libraries only. Installing within the scope of a virtual environment is recommended.

## Requirements

Support is set by the main dependencies required by Graphorge: [PyTorch](https://pytorch.org) and [PyG](https://pyg.org/). The stable version of Graphorge requires PyG `2.6.1` and PyTorch `2.4.1`, which is compatible with Python `3.9-3.12` and a variety of hardware and operating systems, see [PyTorch requirements](https://pytorch.org/get-started/locally/). The following installation procedure was tested on Ubuntu `24.04` using Python `3.11`, with both CPU and GPU (cuda).

```{note}
Graphorge may be compatible with more recent releases of PyTorch and PyG but this was not tested.
```

Most requirements will be automatically installed when installing Graphorge. However, PyTorch and PyG often require a manual installation. Installation instructions are detailed in [PyTorch](https://pytorch.org/get-started/locally/) and [PyG](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) documentations. They are summarized here for convenience:

`````{tab-set}

````{tab-item} CUDA

```{code-block} console
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
    pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.1+cu124.html
    pip install torch-sparse -f https://data.pyg.org/whl/torch-2.4.1+cu124.html
    pip install torch_geometric==2.6.1 -f https://data.pyg.org/whl/torch-2.4.1+cu124.html
```

````

````{tab-item} CPU

```{code-block} console
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cpu
    pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.1+cpu.html
    pip install torch-sparse -f https://data.pyg.org/whl/torch-2.4.1+cpu.html
    pip install torch_geometric==2.6.1 -f https://data.pyg.org/whl/torch-2.4.1+cpu.html
```

````

`````

---
# Package manager

To install Graphorge, use the supported package manager [pip](https://pypi.org/project/pip/) and simply run `pip install`:

```{code-block} console
    pip install Graphorge
```

---
# From source

To install Graphorge from source, clone the related Github repository:

```{code-block} console
    git clone git@github.com:bessagroup/graphorge.git
```

Then, `pip install` from the newly created directory (see [pip documentation](https://pip.pypa.io/en/stable/topics/local-project-installs/#regular-installs) for details):

```{code-block} console
    pip install .
```

````{note}
Consider performing an [editable install](https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs) when contributing to Graphorge:

```{code-block} console
    pip install -e .
```

````


