import copy
import os.path
import sys
from argparse import ArgumentParser


import importlib

from PROXIMAL2.proximal_functions.Common2 import ProximalException, ExtractPairs
from PROXIMAL2.proximal_functions.Operators2 import GenerateOperators
from PROXIMAL2.proximal_functions.Products2 import GenerateProducts
from PROXIMAL2.proximal_functions.GenerateMolFiles2 import GenerateMolFiles

from utils import get_compound_mass_dict, calculate_delta_mass_operator, add_delta_mass, select_operators_by_mass, SaveInputFiles_BAM
import json
import rdkit.Chem.Descriptors

from analyzeCandidates import analyze_cands, process_cands_for_SOM, generate_kcf, analyze_ground_truth_cands, process_ground_truth_cands_for_SOM, generate_kcf_ground_truth

from os.path import isfile, isdir
from os import mkdir, makedirs, listdir
import pickle
import pandas as pd
import time
import shutil
import tqdm
from rdkit import Chem
import csv

margin = 0.5

parser = ArgumentParser()
parser.add_argument('--moi', default="./data/input/GMN_validation_dataset.csv")
parser.add_argument('--som_dir', default='./GNN-SOM-master/')
parser.add_argument('--write_dir', default='./data/output/')
parser.add_argument('--rxn_mets', default="./data/reactions/RetroRules/RetroRules_metabolite_input.csv")
parser.add_argument('--rxns', default="./data/reactions/RetroRules/RetroRules_reactions.csv")
parser.add_argument('--op_dir', default='./data/reactions/RetroRules/operators')
parser.add_argument('--rxn_dir', default='./data/reactions/RetroRules/')
args = vars(parser.parse_args())

print(args)

molecules_of_interest = pd.read_csv(args['moi'])
metabolites = pd.read_csv(args['rxn_mets'])
reaction_list = pd.read_csv(args['rxns'])

OP_CACHE_DIRECTORY = args['op_dir']
WRITE_DIRECTORY = args['write_dir']
path_finalReactions = args['rxn_dir']
SOM_DIRECTORY = args['som_dir']

OUTPUT_DIRECTORY = WRITE_DIRECTORY + 'products'
ground_truth = False
if 'suspect_inchi' in molecules_of_interest.columns or 'suspect_smiles' in molecules_of_interest.columns:
    ground_truth = True


#edit molecules_of_interest for correct use with PROXIMAL2
molecules_of_interest['anchor_name'] = molecules_of_interest['name']
molecules_of_interest['name'] = molecules_of_interest['ID']

def runPROXIMAL2(ground_truth):
    start_time = time.time()
    files = listdir(OP_CACHE_DIRECTORY)
    # organize operators by delta mass
    # find all dictionary keys within the margin of the delta mass
    mass_keys = list(delta_masses.keys())
    mass_keys.sort()

    # added for caching
    if not os.path.isdir(WRITE_DIRECTORY + 'anchor_input_lists'):
        mkdir(WRITE_DIRECTORY + 'anchor_input_lists')
    for idx in molecules_of_interest.index:

        targetMoleculeDir = OUTPUT_DIRECTORY + '/' + \
                        molecules_of_interest['ID'][idx] + '_' + str(molecules_of_interest['suspect_ID'][idx])
        if isdir(targetMoleculeDir):
            shutil.rmtree(targetMoleculeDir)
        mkdir(targetMoleculeDir)
        # change mass to pull the delta from the file
        mass = molecules_of_interest['delta_mass'][idx]
        mass_low_lim = molecules_of_interest['mass'][idx] + float(mass) - margin - 0.5
        mass_up_lim = molecules_of_interest['mass'][idx] + float(mass) + margin + 0.5
        ops = select_operators_by_mass(mass, margin, mass_keys, delta_masses)

        for pos_file in range(len(ops)):

            print("Product: " + str(idx) + '/' + str(len(molecules_of_interest.index) - 1) + "\n" +
                  "Operator: " + str(pos_file) + '/' + str(len(ops) - 1))
            opFilename = OP_CACHE_DIRECTORY + '/' + ops[pos_file] + '.dat'
            f = open(opFilename, 'rb')
            operators = pickle.load(f)['operators']
            f.close()
            f = open(opFilename, 'rb')
            charge_Template = pickle.load(f)['charge_Template']
            f.close()

            rxn_id = ops[pos_file]
            try:
                inputList, inputStructure, inputListNumbering, products, charge = \
                    GenerateProducts(molecules_of_interest['ID'][idx],
                                     molecules_of_interest, opFilename, metabolites)
                temp_filename = WRITE_DIRECTORY + 'anchor_input_lists/' + molecules_of_interest['ID'][idx].replace(":", "_") + '.json'
                SaveInputFiles_BAM(inputList, inputStructure, inputListNumbering, charge, temp_filename)
            except ProximalException as e:
                print('%s -> %s (%d ops): %s' % (molecules_of_interest['ID'][idx],
                                                 rxn_id, len(operators), str(e)))
                continue

            if len(products) >= 1:

                targetReactionDir = targetMoleculeDir + '/' + rxn_id

                mkdir(targetReactionDir)

                try:
                    GenerateMolFiles(targetReactionDir, inputStructure, inputListNumbering, products,
                                     charge, charge_Template, molecules_of_interest.loc[idx,])

                except ProximalException as e:
                    print(str(e))
                    continue

                for prod in listdir(targetReactionDir):
                    f = open(targetReactionDir + '/' + prod)
                    data = json.load(f)
                    prod_j = data['GeneratedProduct'][0]
                    prod_mol = Chem.MolFromMolBlock(prod_j['mol'])
                    if prod_mol is not None:
                        mass_prod = Chem.Descriptors.ExactMolWt(prod_mol)
                        if (mass_prod >= mass_low_lim) and (mass_prod <= mass_up_lim):
                            prod_inchikey = Chem.MolToInchiKey(prod_mol)
                            os.rename(targetReactionDir + '/' + prod, targetReactionDir + '/' + prod_inchikey + prod)
                        else:
                            os.remove(targetReactionDir + '/' + prod)
                    else:
                        os.remove(targetReactionDir + '/' + prod)

                print('%s -> %s (%d ops) => %d products' %
                      (molecules_of_interest['ID'][idx], rxn_id,
                       len(operators), len(listdir(targetReactionDir))))

                if len(listdir(targetReactionDir)) == 0:
                    shutil.rmtree(targetReactionDir)

        if len(listdir(targetMoleculeDir)) == 0:
            shutil.rmtree(targetMoleculeDir)

    print("--- %s seconds = %s hours ---" % ((time.time() - start_time), (time.time() - start_time) / 60 / 60))
    return

if not isdir(OP_CACHE_DIRECTORY):
    makedirs(OP_CACHE_DIRECTORY)
if not isdir(OUTPUT_DIRECTORY):
    makedirs(OUTPUT_DIRECTORY)

# Remove redundancy among pairs
if not isfile(path_finalReactions+"FinalMetabolicSpace_prova.pkl"):
    reactions = ExtractPairs(reaction_list,metabolites)
    reactions = add_delta_mass(reactions,metabolites)
    with open(path_finalReactions+"FinalMetabolicSpace_prova.pkl", "wb") as f:
        pickle.dump(reactions, f)
else:
    with open(path_finalReactions+"FinalMetabolicSpace_prova.pkl", 'rb') as f: 
        reactions = pickle.load(f)

start_time = time.time()
delta_masses = {}
for i in reactions.index:
    opFilename = OP_CACHE_DIRECTORY + '/' + reactions.id[i] + '.dat'
    print('Building operator: '+str(i)+'/'+str(len(reactions.index)-1))
    if not isfile(opFilename):
        print('Building operator for %s...' % reactions.id[i])
        try:
            operators, operatorsMainAtom, operatorsMainAtomPos, charge_Template = \
                    GenerateOperators(reactions.loc[i,], opFilename, metabolites)
        except ProximalException as e:
            print(str(e))
            continue
        except ValueError as ve:
        	print(str(ve))
        	continue
    if isfile(opFilename):
        delta = reactions.deltaMass[i][0]
        if delta in delta_masses.keys():
            delta_masses[delta].append(reactions.id[i])
        else:
            delta_masses[delta] = [reactions.id[i]]


print("--- %s seconds = %s hours ---" % ((time.time() - start_time),(time.time() - start_time)/60/60))

runPROXIMAL2(ground_truth)

molecules_of_interest['name'] = molecules_of_interest['anchor_name']

if ground_truth:
    total_results = analyze_ground_truth_cands(WRITE_DIRECTORY, molecules_of_interest)
    filtered_results = process_ground_truth_cands_for_SOM(reactions, total_results, SOM_DIRECTORY, WRITE_DIRECTORY)
    generate_kcf_ground_truth(WRITE_DIRECTORY, molecules_of_interest, filtered_results)
else:
    total_results = analyze_cands(WRITE_DIRECTORY, molecules_of_interest)
    filtered_results = process_cands_for_SOM(reactions, total_results, SOM_DIRECTORY, WRITE_DIRECTORY)
    generate_kcf(WRITE_DIRECTORY, molecules_of_interest, filtered_results)

