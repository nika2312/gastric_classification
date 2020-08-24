# Classification of gastric cancer subtypes using graph-CNN

Code for the Deep Learning course project in Tel-Aviv University, 2018.

The method is inspired by [Rhee et al. 2017](https://arxiv.org/pdf/1711.05859.pdf), who presented a breast cancer subtypes classification, also using gene expression information from the TCGA project. In addition, the STRING database, which contains network interactions between proteins in the form of a weighted graph, is used to model protein interactions as a knowledge base.
The gCNN implementation is taken from [Defferrard et al., 2016](http://papers.nips.cc/paper/6081-convolutional-neural-networks-on-graphs-with-fast-localized-spectral-filtering.pdf).

*prepare_data.py* contains the dataset building from raw gene expression files downloaded from the TCGA website, and their alignment with the molecular study labels.
*build_graph.py* handles the creation of the protein-protein interaction graph from the data downloaded from the STRING website.
The *“gcnn”* directory contains the graph-CNN model. The *“lib”* directory contains the model’s implementation as published by the authors of Defferrard et al. 2016:
*coarsening.py* contains the graph coarsening using Graclus and Metis algorithms. *graph.py* contains the laplacian calculation. *models.py* contains the actual deep learning models. 
*run_model.py* runs the training process and outputs the results.
