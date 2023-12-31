molecules_of_interest = "../data/input/GMN_validation_dataset.csv"
SOM_DIRECTORY = '../GNN-SOM-master/'
WRITE_DIRECTORY = '../data/output/'

######KEGG biotransformations
#metabolites_list = "../data/reactions/KEGG/KEGG_metabolite_input.csv"
#reaction_list = "../data/reactions/KEGG/KEGG_reactions.csv"
#OP_CACHE_DIRECTORY = '../data/reactions/KEGG/operators'
#path_finalReactions = '../data/reactions/KEGG/'

######RetroRules biotransformations
metabolites_list = "../data/reactions/RetroRules/RetroRules_metabolite_input.csv"
reaction_list = "../data/reactions/RetroRules/RetroRules_reactions.csv"
OP_CACHE_DIRECTORY = '../data/reactions/RetroRules/operators'
path_finalReactions = '../data/reactions/RetroRules/'

conda activate proximal2
python runPROX2.py --moi molecules_of_interest --som_dir SOM_DIRECTORY --write_dir WRITE_DIRECTORY --rxn_mets metabolites_list --rxns reaction_list --op_dir OP_CACHE_DIRECTORY --rxn_dir path_finalReactions
conda deactivate

conda activate som
python runSOM.py --moi molecules_of_interest --som_dir SOM_DIRECTORY --write_dir WRITE_DIRECTORY --rxn_dir path_finalReactions
conda deactivate


