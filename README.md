# Molecular structure discovery from untargeted metabolomics data using biotransformation rules and global molecular networking

This repository contains code to implement the biotransformation-based annotation method (**BAM**), as well as the data used to validate BAM [[1]](#1).

## Installation and Requirements

BAM uses previous tools, PROXIMAL2 (https://github.com/HassounLab/PROXIMAL2) [[2]](#2) and GNN-SOM (https://github.com/HassounLab/GNN-SOM) [[3]](#3). To use BAM, these tools need to be downloaded and included under the "BAM-main" directory. Also, both conda environments need to be created as described in those repositories and named "proximal2" and "som" respectively.

## Usage

All input information required is specified in runBAM.sh.

## References 
<a id="1">[1]</a> 
Bittremieux, W., Avalon, N. E., Thomas, S. P., Kakhkhorov, S. A., Aksenov, A. A., Gomes, P. W. P., ... & Dorrestein, P. C. (2022). Open access repository-scale propagated nearest neighbor suspect spectral library for untargeted metabolomics. BioRxiv, 2022-05.

<a id="2">[2]</a> 
Balzerani, F., Blasco, T., Perez, S., Valcarcel, L. V., Planes, F. J., & Hassoun, S. (2023). Extending PROXIMAL to predict degradation pathways of phenolic compounds in the human gut microbiota. bioRxiv, 2023-05.

<a id="3">[3]</a> 
Porokhin, V., Liu, L. P., & Hassoun, S. (2023). Using graph neural networks for site-of-metabolism prediction and its applications to ranking promiscuous enzymatic products. Bioinformatics, 39(3), btad089.