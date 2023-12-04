import csv
import copy
import json
from os.path import isfile
import os
from os import mkdir

import numpy as np
import torch
from torch_geometric.data import Data
from rdkit.Chem import Draw
from rdkit.Chem.Draw.MolDrawing import DrawingOptions
import tqdm


import requests
import pandas as pd
import pickle
from collections import Counter
from rdkit import Chem
import json
import pandas as pd
import time
import sys
csv.field_size_limit(sys.maxsize)
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument('--moi', default="../data/input/GMN_validation_dataset.csv")
parser.add_argument('--write_dir', default='../data/output/')
parser.add_argument('--rxn_dir', default='../data/reactions/RetroRules/')
parser.add_argument('--som_dir', default='../GNN-SOM-master/')
args = vars(parser.parse_args())

WRITE_DIRECTORY = args['write_dir']
path_finalReactions = args['rxn_dir']
SOM_DIRECTORY = args['som_dir']

kcf_path = WRITE_DIRECTORY+'kcf/'

with open('../data/input/GMN_validation_dataset.csv', 'r') as f:
    inputs = list(csv.reader(f, delimiter=","))
col_names = inputs[0]
inputs = inputs[1:]


with open(path_finalReactions+"FinalMetabolicSpace_prova.pkl", 'rb') as f:
    reactions = pickle.load(f)

with open(WRITE_DIRECTORY + 'filtered_cands_for_ranking.csv', 'r') as f:
    results = list(csv.reader(f, delimiter=","))
results = results[1:]

ground_truth = False
if 'suspect_inchi' in col_names or 'suspect_smiles' in col_names:
    ground_truth = True


import importlib
gnn_som = importlib.import_module("../GNN-SOM-master")
from gnn_som import createGnnSom, loadGnnSomState
from gnn_som.MolFromKcf import MolFromKcfFile

# from GNN_SOM_functions import construct_graph, predict

def construct_graph(config, mol, enzyme):
    numFeatures = sum(len(feature) for feature in config['features'].values())
    x = torch.zeros((mol.GetNumAtoms(), numFeatures), dtype=torch.float32)
    for atom in mol.GetAtoms():
        x[atom.GetIdx(), config['features']['enzyme'].index(enzyme)] = 1
        offset = len(config['features']['enzyme'])
        x[atom.GetIdx(), offset + config['features']['element'].index(atom.GetSymbol())] = 1
        offset += len(config['features']['element'])
        x[atom.GetIdx(), offset + config['features']['kcfType'].index(atom.GetProp('kcfType'))] = 1

    edgeIndex = torch.zeros((2, mol.GetNumBonds() * 2), dtype=torch.int64)
    for bond in mol.GetBonds():
        i = bond.GetIdx()
        edgeIndex[0][i * 2] = bond.GetBeginAtomIdx()
        edgeIndex[1][i * 2] = bond.GetEndAtomIdx()
        edgeIndex[0][i * 2 + 1] = bond.GetEndAtomIdx()
        edgeIndex[1][i * 2 + 1] = bond.GetBeginAtomIdx()

    data = Data(x=x, edgeIndex=edgeIndex)
    return data

def predict(models, data):
    y = None
    for model in models:
        newY = torch.sigmoid(model(data.x, data.edgeIndex))
        y = newY if y is None else torch.add(y, newY)
    y = torch.div(y, len(models))
    return y

def GNN_SOM_models():
    with open(SOM_DIRECTORY + 'data/config.json', 'r') as f:
        config = json.load(f)
    config['features']['enzyme'] = [tuple(ec) for ec in config['features']['enzyme']]

    models = []
    for i, params in enumerate(config['models']):
        model = createGnnSom(*config['models'][i])
        loadGnnSomState(model, torch.load(SOM_DIRECTORY + 'data/model%d.pt' % i, map_location=torch.device('cpu')))
        models.append(model)

    return config, models


def applySOM(config, models):
    start_time = time.time()
    to_save = []
    for result in tqdm.tqdm(results, desc="running GNN-SOM", unit='pair'):

        if result[0].count('_') == 1:
            comps = result[0].split('_')
            sub = comps[0]
            prod = comps[1]
        elif result[0].count('_') == 3:
            comps = result[0].split('_')
            sub = comps[0] + '_' + comps[1]
            prod = comps[2] + '_' + comps[3]
        else:
            print('error')

        mol = MolFromKcfFile(kcf_path + sub + '.kcf')

        temp_enzymes = result[-1][2:-2]
        temp_enzymes = temp_enzymes.split('), (')
        enzymes = []
        for enz in temp_enzymes:
            temp = enz.split(', ')
            temp = [int(i) for i in temp]
            enzymes.append(tuple(temp))

        # run model for each enzyme
        for enzyme in enzymes:
            data = construct_graph(config, mol, enzyme)

            y = predict(models, data)

            to_save.append([result[0], enzyme, y.tolist()])

    print("--- %s seconds = %s hours ---" % ((time.time() - start_time), (time.time() - start_time) / 60 / 60))

    return to_save

def rank_SOM_validation(gnn_som_df):
    save_gnnsom = []
    for result in tqdm.tqdm(results, desc="ranking", unit='pair'):
        true_ik = result[10]
        block1_trueik = true_ik.split('-')[0]

        # get all inchikey info
        all_ik = result[11]
        all_ik = all_ik[2:-2]
        all_ik = all_ik.split("', '")
        ik_dict = {}
        for ik in all_ik:
            # if ik != '':
            temp = ik.split('-')[0]
            if temp in ik_dict.keys():
                ik_dict[temp] += 1
            else:
                ik_dict[temp] = 1
        sort_iks = sorted(ik_dict.items(), key=lambda x: x[1])

        # get all operator info
        operators = result[13]
        operators = operators[2:-2]
        operators = operators.split("', '")
        assert len(operators) == len(all_ik)
        op_dict = Counter(operators)
        sort_ops = sorted(op_dict.items(), key=lambda x: x[1])

        # substrate and product name
        if result[0].count('_') == 1:
            comps = result[0].split('_')
            sub = comps[0]
            prod = comps[1]
        elif result[0].count('_') == 3:
            comps = result[0].split('_')
            sub = comps[0] + '_' + comps[1]
            prod = comps[2] + '_' + comps[3]
        else:
            print('error')

        # have to find the gnn-som likelihood of each product
        temp_df = gnn_som_df.loc[gnn_som_df['pair'] == result[0]]
        som_results = {}
        filename_dict = {}
        for op in op_dict.keys():
            idx = reactions.index[reactions['id'] == op].tolist()[0]
            EC_list = reactions.iloc[idx]['EC']
            enzymes = []
            for enz in EC_list:
                temp = enz.split('.')[0:2]
                if '-' in temp:
                    if '-' == temp[1]:
                        temp = [t for t in config['features']['enzyme'] if t[0] == int(temp[0])]
                        for t in temp:
                            if t not in enzymes and t in config['features']['enzyme']:
                                enzymes.append(t)
                    else:
                        print('error')
                else:
                    temp = [int(i) for i in temp]
                    if tuple(temp) not in enzymes and tuple(temp) in config['features']['enzyme']:
                        enzymes.append(tuple(temp))

            gen_prods = os.listdir(WRITE_DIRECTORY + 'products/' + result[0] + '/' + op)
            for gen_prod in gen_prods:
                f = open(WRITE_DIRECTORY + 'products/' + result[0] + '/' + op + '/' + gen_prod)
                json_prod = json.load(f)
                indices = json_prod['ObtainedFromInput']

                f = open(WRITE_DIRECTORY + 'anchor_input_lists/' + sub.replace(':', '_') + '.json')
                inputList = json.load(f)

                reaction_center = []
                input_index = []
                for index in indices:
                    reaction_center.append(inputList['inputList'][index - 1])
                    input_index.append(inputList['inputListNumbering'][index - 1])

                true_index = []
                for index in input_index:
                    true_index.append(index[0] - 1)

                assert len(json_prod['GeneratedProduct']) == 1
                prod_j = json_prod['GeneratedProduct'][0]
                prod_mol = Chem.MolFromMolBlock(prod_j['mol'])
                if prod_mol is not None:
                    ik = Chem.MolToInchiKey(prod_mol).split('-')[0]
                    if ik in ik_dict.keys():
                        max_y = 0
                        for enz in enzymes:
                            y = temp_df.loc[temp_df['EC'] == enz]['y'].to_list()[0]
                            for index in true_index:
                                temp_y = round(y[index][0], 2)
                                if temp_y > max_y:
                                    max_y = temp_y

                        if ik not in som_results.keys():
                            som_results[ik] = max_y
                            filename_dict[ik] = WRITE_DIRECTORY + 'products/' + result[0] + '/' + op + '/' + gen_prod
                        else:
                            if som_results[ik] < max_y:
                                som_results[ik] = max_y

        sort_som = sorted(som_results.items(), key=lambda x: x[1])

        previous_cands = []
        j = -1
        top = []
        while block1_trueik not in top:
            previous_cands.extend(top)
            top = [sort_som[j][0]]
            #resave products under ranking and make sure to save all the ones where only one candidate was generated
            top_val = sort_som[j][1]
            j -= 1
            if len(sort_som) >= abs(j):
                while sort_som[j][1] == top_val:
                    top.append(sort_som[j][0])
                    j -= 1
                    if abs(j) > len(sort_som):
                        break

        previous_cands = set(previous_cands)
        ranks = range(len(previous_cands) + 1, len(previous_cands) + len(top) + 1)
        avg_rank = sum(ranks) / len(top)
        if len(previous_cands) == 0:
            top_rank = 1
        else:
            top_rank = 0

        if not os.path.isdir(WRITE_DIRECTORY + 'ranked_derivatives/' + result[0]):
            mkdir(WRITE_DIRECTORY + 'ranked_derivatives/' + result[0])
        sort_som.reverse()
        top_val = -1
        count = 0
        for j in range(0, len(sort_som)):
            if sort_som[j][1] != top_val:
                count += 1
            f = open(filename_dict[sort_som[j][0]])
            json_prod = json.load(f)
            new_filename = WRITE_DIRECTORY + 'ranked_derivatives/' + result[0] + '/' + str(count) + '_' + sort_som[j][0] + '.json'
            with open(new_filename, 'w') as f:
                json.dump(json_prod, f, indent=2)

        save_gnnsom.append([result[0], top_rank, avg_rank, len(top)])

    return save_gnnsom

def rank_SOM(gnn_som_df):
    for result in tqdm.tqdm(results, desc="ranking", unit='pair'):
        # get all inchikey info
        all_ik = result[11]
        all_ik = all_ik[2:-2]
        all_ik = all_ik.split("', '")
        ik_dict = {}
        for ik in all_ik:
            temp = ik.split('-')[0]
            if temp in ik_dict.keys():
                ik_dict[temp] += 1
            else:
                ik_dict[temp] = 1
        sort_iks = sorted(ik_dict.items(), key=lambda x: x[1])

        # get all operator info
        operators = result[13]
        operators = operators[2:-2]
        operators = operators.split("', '")
        assert len(operators) == len(all_ik)
        op_dict = Counter(operators)
        sort_ops = sorted(op_dict.items(), key=lambda x: x[1])

        # substrate and product name
        if result[0].count('_') == 1:
            comps = result[0].split('_')
            sub = comps[0]
            prod = comps[1]
        elif result[0].count('_') == 3:
            comps = result[0].split('_')
            sub = comps[0] + '_' + comps[1]
            prod = comps[2] + '_' + comps[3]
        else:
            print('error')

        # have to find the gnn-som likelihood of each product
        temp_df = gnn_som_df.loc[gnn_som_df['pair'] == result[0]]
        som_results = {}
        filename_dict = {}
        for op in op_dict.keys():
            idx = reactions.index[reactions['id'] == op].tolist()[0]
            EC_list = reactions.iloc[idx]['EC']
            enzymes = []
            for enz in EC_list:
                temp = enz.split('.')[0:2]
                if '-' in temp:
                    if '-' == temp[1]:
                        temp = [t for t in config['features']['enzyme'] if t[0] == int(temp[0])]
                        for t in temp:
                            if t not in enzymes and t in config['features']['enzyme']:
                                enzymes.append(t)
                    else:
                        print('error')
                else:
                    temp = [int(i) for i in temp]
                    if tuple(temp) not in enzymes and tuple(temp) in config['features']['enzyme']:
                        enzymes.append(tuple(temp))

            gen_prods = os.listdir(WRITE_DIRECTORY + 'products/' + result[0] + '/' + op)
            for gen_prod in gen_prods:
                f = open(WRITE_DIRECTORY + 'products/' + result[0] + '/' + op + '/' + gen_prod)
                json_prod = json.load(f)
                indices = json_prod['ObtainedFromInput']

                f = open(WRITE_DIRECTORY + 'anchor_input_lists/' + sub.replace(':', '_') + '.json')
                inputList = json.load(f)

                reaction_center = []
                input_index = []
                for index in indices:
                    reaction_center.append(inputList['inputList'][index - 1])
                    input_index.append(inputList['inputListNumbering'][index - 1])

                true_index = []
                for index in input_index:
                    true_index.append(index[0] - 1)

                assert len(json_prod['GeneratedProduct']) == 1
                prod_j = json_prod['GeneratedProduct'][0]
                prod_mol = Chem.MolFromMolBlock(prod_j['mol'])
                if prod_mol is not None:
                    ik = Chem.MolToInchiKey(prod_mol).split('-')[0]
                    if ik in ik_dict.keys():
                        max_y = 0
                        for enz in enzymes:
                            y = temp_df.loc[temp_df['EC'] == enz]['y'].to_list()[0]
                            for index in true_index:
                                temp_y = round(y[index][0], 2)
                                if temp_y > max_y:
                                    max_y = temp_y

                        if ik not in som_results.keys():
                            som_results[ik] = max_y
                            filename_dict[ik] = WRITE_DIRECTORY + 'products/' + result[0] + '/' + op + '/' + gen_prod
                        else:
                            if som_results[ik] < max_y:
                                som_results[ik] = max_y

        sort_som = sorted(som_results.items(), key=lambda x: x[1])


        if not os.path.isdir(WRITE_DIRECTORY + 'ranked_derivatives/' + result[0]):
            mkdir(WRITE_DIRECTORY + 'ranked_derivatives/' + result[0])
        sort_som.reverse()
        top_val = -1
        count = 0
        for j in range(0, len(sort_som)):
            if sort_som[j][1] != top_val:
                count += 1
            f = open(filename_dict[sort_som[j][0]])
            json_prod = json.load(f)
            new_filename = WRITE_DIRECTORY + 'ranked_derivatives/' + result[0] + '/' + str(count) + '_' + sort_som[j][0] + '.json'
            with open(new_filename, 'w') as f:
                json.dump(json_prod, f, indent=2)

    return

if __name__ == '__main__':
    if not os.path.isdir(WRITE_DIRECTORY + 'ranked_derivatives'):
        mkdir(WRITE_DIRECTORY + 'ranked_derivatives')
    config, models = GNN_SOM_models()
    output = applySOM(config, models)
    gnn_som_df = pd.DataFrame(output, columns=['pair', 'EC', 'y'])

    if ground_truth:
        gnn_som_result = rank_SOM_validation(gnn_som_df)
    else:
        rank_SOM(gnn_som_df)

