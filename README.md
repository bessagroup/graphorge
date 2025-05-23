<p align="center">
  <a href=""><img alt="logo" src="https://github.com/user-attachments/assets/95fd5e9b-fdc2-4adc-a0a1-f47c9ccd3216" width="80%"></a>
</p>

# What is Graphorge?

[**Docs**](https://)
| [**Installation**](https://)
| [**GitHub**](https://)

### Summary

Graphorge is an open-source Python package built on [PyTorch](https://pytorch.org/) that streamlines the development and evaluation of graph neural networks. It provides a complete workflow encompassing data preprocessing, dataset management, model training, prediction, and result post-processing. Graphorge includes a fully implemented, highly customizable example architecture to demonstrate practical use, and its code is thoroughly documented and commented, making it especially accessible to researchers new to implementing graph neural networks. Although originally developed for computational mechanics, Graphorge’s core functionality is general and can be applied across a wide range of domains.

### Statement of need

Graph neural networks have emerged as a powerful modeling tool for data with relational or geometric structure. Existing graph neural network libraries tend to fall into two categories. On the one hand, many research-specific implementations are minimally documented and tightly tailored to particular benchmarks. While this may enable results reproducibility, code comprehension is often limited, as well as its customization and generalization to different applications. On the other hand, general purpose frameworks such as [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/) and [Deep Graph Library](https://www.dgl.ai/) offer robust, high-performance platforms to implement graph neural networks, usually providing multiple backend support and integration of state-of-the-art scientific contributions.

Graphorge is not intended as an alternative to general-purpose frameworks, which are well-suited to the needs of advanced users. Instead, it is designed as a practical and educational tool for researchers and students that aim to understand and be able to implement a full Graph Neural Network pipeline. While only including a single, highly customizable architecture to demonstrate practical use, every single module in Graphorge is extensively documented and commented, aiming to maximize code comprehension. Moreover, rather than abstracting the workflow behind opaque interfaces, Graphorge provides deliberatly fully functional, modular scripts for each stage -- from pre-processing and dataset handling to model training and post-processing -- making it an ideal extensible, starting point for those looking to implement or customize Graph Neural Networks in a research environment.

### Authorship & Citation
Graphorge was originally developed by Bernardo P. Ferreira<sup>[1](#f1)</sup>.

<sup id="f1"> 1 </sup> Profile: [LinkedIN](https://www.linkedin.com/in/bpferreira/), [ORCID](https://orcid.org/0000-0001-5956-3877), [ResearchGate](https://www.researchgate.net/profile/Bernardo-Ferreira-11?ev=hdr_xprf)

----

# Getting started

For an overview of the package, how to install it, and several documented application examples, please refer to [Graphorge's documentation]()!

# Community Support

If you find any **issues**, **bugs** or **problems** with Graphorge, please use the [GitHub issue tracker](https://github.com/BernardoFerreira/graphorge/issues) to report them. Provide a clear description of the problem, as well as a complete report on the underlying details, so that it can be easily reproduced and (hopefully) fixed!

You are also welcome to post there any **questions**, **comments** or **suggestions** for improvement in the [GitHub discussions](https://github.com/BernardoFerreira/graphorge/discussions) space!

Please refer to Graphorge's [Code of Conduct](https://github.com/BernardoFerreira/graphorge/blob/master/CODE_OF_CONDUCT.md).


# Credits

* Graphorge was originally inspired by [Google DeepMind](https://github.com/google-deepmind/deepmind-research), namely the contribution *Learning Mesh-Based Simulation with Graph Networks (2021)* by Pfaff and coworkers ([paper](https://arxiv.org/abs/2010.03409), [code](https://github.com/google-deepmind/deepmind-research/tree/master/learning_to_simulate)).

* Bernardo P. Ferreira is thankful to [Guillaume Broggi](https://github.com/GuillaumeBroggi) for his essential contribution to developing comprehensive benchmarks that illustrate Graphorge's workflow and support the open-source project.

* Bernardo P. Ferreira acknowledges the contribution of [Rui Pinto](https://github.com/ruibmpinto), whose work enabled the seamless handling of time series data.

# License

Copyright 2023, Bernardo Ferreira

All rights reserved.

Graphorge is a free and open-source software published under a MIT License.
