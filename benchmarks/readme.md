# Graphorge example directory

## Introduction

The Graphorge example directory provides examples introducing basic concepts required to use Graphorge and demonstrating its usage in a research context.

## Instructions

The examples are provided as Jupyter notebooks to enhance clarity and support interactive, didactic learning, although Graphorge would probably be used in a non-interactive setting when going to production.

The provided examples can be ran:
- In the cloud: each compatible example features a [Run this notebook in Google Colab](https://colab.research.google.com) link at the top, which opens a Jupyter notebook in Google Colab. The torch version available in Google Colab might not be compatible with Graphorge. If this is the case, see [PyTorch documentation](https://docs.pytorch.org/tutorials/beginner/colab.html) for instructions.
- Locally: this option requires to setup a local installation of Graphorge, refer to the installation instructions.

## List of examples

|Example|Code|Field|Concepts covered|
| --- | --- | --- | --- |
|[Shell buckling](./mechanics/gnn_shell_buckling/gnn_shell_buckling)|[Github](./mechanics/gnn_shell_buckling)|Mechanics|Graphorge basics, regression on a global target.|
|[Permeability of a porous medium](./cfd/gnn_porous_medium/gnn_porous_medium)|[Github](./gnn_porous_medium/gnn_porous_medium)|CFD|Graphorge basics, regression on a global target.|