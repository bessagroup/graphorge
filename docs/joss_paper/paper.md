---
title: 'Graphorge: An open-source forge of graph neural networks'
tags:
  - Python
  - machine learning
  - graph neural networks
  - computational mechanics

authors:
  - name: Bernardo P. Ferreira
    orcid: 0000-0001-5956-3877
    affiliation: 1
  - name: Guillaume Broggi
    orcid: 0000-0001-6001-6328
    affiliation: 1
  - name: Miguel A. Bessa
    orcid: 0000-0002-6216-0355
    corresponding: true
    affiliation: 1
affiliations:
 - name: School of Engineering, Brown University, United States of America
   index: 1
date: 15 June 2025
bibliography: references.bib

---

# Summary

[Graphorge](https://github.com/bessagroup/graphorge) is an open-source Python package built on [PyTorch](https://pytorch.org/) that streamlines the development and evaluation of graph neural networks. It provides a complete workflow encompassing data pre-processing, dataset management, model training, prediction, and result post-processing. Graphorge includes a fully implemented, highly customizable example architecture to demonstrate practical use, and its code is thoroughly documented and commented, making it especially accessible to researchers new to implementing graph neural networks. Although originally developed for computational mechanics, Graphorge’s core functionality is general and can be applied across a wide range of domains.

![Logo of [Graphorge](https://github.com/bessagroup/graphorge). \label{fig:graphorge_logo_horizontal_white}](graphorge_logo_horizontal_white.png)

# Statement of need

Graph neural networks have emerged as a powerful modeling tool for data with relational or geometric structure. Existing graph neural network libraries tend to fall into two categories. On the one hand, many research-specific implementations are minimally documented and tightly tailored to particular benchmarks. While this may enable results reproducibility, code comprehension is often limited, as well as its customization and generalization to different applications. On the other hand, general purpose frameworks such as [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/) [@fey:2019a] and [Deep Graph Library](https://www.dgl.ai/) [@wang:2020a] offer robust, high-performance platforms to implement graph neural networks, usually providing multiple backend support and integration of state-of-the-art scientific contributions.

Graphorge is not intended as an alternative to general-purpose frameworks. Instead, it is designed as a practical and educational tool for researchers and students that aim to understand and be able to implement a full Graph Neural Network pipeline. While only including a single, highly customizable architecture [@battaglia:2018a] to demonstrate practical use, every single module in Graphorge is extensively documented and commented, aiming to maximize code comprehension. Moreover, rather than abstracting the workflow behind opaque interfaces, Graphorge provides deliberatly fully functional, modular scripts for each stage -- from pre-processing and dataset handling to model training and post-processing -- making it an ideal extensible, starting point for those looking to implement or customize graph neural networks in a research environment.

![Example of Graph Neural Network model with an encoder-processor-decoder architecture.](graphorge_overview.png)

Graphorge is built on the graph neural network conceptual framework established by Battaglia and coworkers [@battaglia:2018a]. In particular, it is originally inspired by the contribution of Sanchez-Gonzalez and coworkers [@sanchez:2020a], namely the [supporting code](https://github.com/google-deepmind/deepmind-research/tree/master/learning_to_simulate) made available through [Google DeepMind](https://github.com/google-deepmind/deepmind-research). In a similar scope, it is worth mentioning [meshgraphnets](https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets) by Pfaff and coworkers [@pfaff:2021a], [gns](https://github.com/geoelements/gns) by Kumar and Vantassel [@kumar:2023a], and [physicsnemo](https://github.com/NVIDIA/physicsnemo?tab=readme-ov-file) by NVIDIA.


# Acknowledgements

Miguel A. Bessa acknowledges the support provided from...

# References
