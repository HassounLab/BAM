# Molecular structure discovery from untargeted metabolomics data using biotransformation rules and global molecular networking

This repository contains code to implement the biotransformation-based annotation method (**BAM**) [[1]](#1), as well as the data used to validate BAM [[2]](#2).

## Installation and Requirements

BAM uses previous tools, PROXIMAL2 (https://github.com/HassounLab/PROXIMAL2) [[3]](#3) and GNN-SOM (https://github.com/HassounLab/GNN-SOM) [[4]](#4). To use BAM, these tools need to be downloaded and included under the "BAM-main" directory. Also, both conda environments need to be created as described in those repositories and named "proximal2" and "som" respectively.

## Usage

All input information required is specified in runBAM.sh.

The desired reaction dataset of interest needs to be specified. We have used KEGG and RetroRules as examples. 

- metabolites_list = csv file of metabolites and their structures (represented by SMILES) that are included in the reaction dataset
-	reaction_list = csv file of the reactions of interest. It must have the following columns: "id", "formula", "EC".
-	OP_CACHE_DIRECTORY = directory location to save the generated operators
-	path_finalReactions = directory location to save the list of metabolite pairs represented by the generated operators

Also, the anchor-suspect pairs of interest need to be specified. We have used a set of pairs derived from the molecular network used to create the suspect library, and this data is set as the default input.

-	molecules_of_interest = csv file of list of queries. It must have an identifier and mass for the suspect as well as anchor and a SMILES that represents the anchor.

Finally, two more directories must be specified. If following these instructions, the provided example input will be sufficient.
-	SOM_DIRECTORY = directory location of GNN-SOM tool downloaded from corresponding github page
-	WRITE_DIRECTORY = directory location of output of BAM


Once the code and conda environments are structured as specified in the Installation and Requirements section  above, run the runBAM.sh file as it is to run the algorithm with the files as described here. Note that no conda environment needs to be activated.
```
sh runBAM.sh
```

## Perform evaluation of BAM using the dataset derived from the Global Molecular Network and reaction data from KEGG and RetroRules

All data necessary to run the evaluation of BAM described in our paper is included in the data folder. All variables in runBAM.sh currently point to KEGG reaction data. As a result, running the code as is will generate and rank candidates using KEGG data for the suspects from the Global Molecular Network usings anchors determined from that network. To change to using RetroRules reaction data, simply comment the “KEGG biotransformations” block and uncomment the “RetroRules biotransformations” box, and then run the runBAM.sh file.

## Perform BAM using another dataset of queries and/or other reaction data 

To apply BAM to another dataset of queries, simply change the specified molecules_of_interest csv file. Again, the only required columns are an identifier and mass for the suspect as well as anchor and a SMILES that represents the anchor. BAM checks if the suspect molecule is known by checking whether the SMILES or InChI is specified in the molecules_of_interest csv file. So, BAM will automatically run an analysis of results if the suspect is known, otherwise it will not perform that analysis. 

To apply BAM using other reaction data, the four reaction variables (metabolites_list, reaction_list, OP_CACHE_DIRECTORY, path_finalReactions) need to be appropriately defined in runBAM.sh. As the only files are metabolites_list and reaction_list, these two files must be in a csv format and have the same columns as specified in the above Usage section.

## References 
<a id="1">[1]</a> 
Martin, M., Bittremieux, W., & Hassoun, S. (2024). Molecular structure discovery for untargeted metabolomics using biotransformation rules and global molecular networking. bioRxiv, 2024-02.

<a id="2">[2]</a> 
Bittremieux, W., Avalon, N. E., Thomas, S. P., Kakhkhorov, S. A., Aksenov, A. A., Gomes, P. W. P., ... & Dorrestein, P. C. (2022). Open access repository-scale propagated nearest neighbor suspect spectral library for untargeted metabolomics. BioRxiv, 2022-05.

<a id="3">[3]</a> 
Balzerani, F., Blasco, T., Perez, S., Valcarcel, L. V., Planes, F. J., & Hassoun, S. (2023). Extending PROXIMAL to predict degradation pathways of phenolic compounds in the human gut microbiota. bioRxiv, 2023-05.

<a id="4">[4]</a> 
Porokhin, V., Liu, L. P., & Hassoun, S. (2023). Using graph neural networks for site-of-metabolism prediction and its applications to ranking promiscuous enzymatic products. Bioinformatics, 39(3), btad089.
