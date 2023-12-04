
import csv
import os.path

from rdkit import Chem
import rdkit.Chem.Descriptors
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem import MACCSkeys
import copy
import numpy as np
import json


from os.path import isfile, isdir
from os import mkdir, makedirs, rmdir, listdir, remove
import pickle
import tqdm
import pandas as pd
from kcfconvoy import KCFvec

from utils import create_img_no_highlights


def analyze_cands(WRITE_DIRECTORY, pairs):
    SVG_DIRECTORY = WRITE_DIRECTORY + 'svgs/'
    OUTPUT_DIRECTORY = WRITE_DIRECTORY + 'products/'
    if not isdir(SVG_DIRECTORY):
        mkdir(SVG_DIRECTORY)

    if not os.path.isdir(WRITE_DIRECTORY + 'ranked_derivatives/'):
        mkdir(WRITE_DIRECTORY + 'ranked_derivatives/')


    stats = []
    stats_numcorrect = []
    stats_numprods = []

    # iterate through the input
    for id, pair in tqdm.tqdm(pairs.iterrows(), desc="Finding Correct Matches", unit="pair"):

        result = pair['ID'] + '_' + str(pair['suspect_ID'])

        target_mass = pair['suspect_mass']

        count = 0
        masses = []
        prod_inchikeys = []
        prod_inchikeys_block1 = []
        prod_smiles = []
        operators = []
        reaction_ids = []

        if os.path.isdir(OUTPUT_DIRECTORY + "/" + result):

            curr_filename = ""
            for op in listdir(OUTPUT_DIRECTORY + '/' + result):
                if op[0] != '.':
                    for prod in listdir(OUTPUT_DIRECTORY + '/' + result + '/' + op):
                        f = open(OUTPUT_DIRECTORY + '/' + result + '/' + op + '/' + prod)
                        data = json.load(f)
                        # data['GeneratedProduct'], data['TemplateReaction'], data['QueryInformation']
                        assert len(data['GeneratedProduct']) == 1
                        prod_j = data['GeneratedProduct'][0]
                        prod_mol = Chem.MolFromMolBlock(prod_j['mol'])
                        if prod_mol is not None:
                            curr_filename = OUTPUT_DIRECTORY + '/' + result + '/' + op + '/' + prod
                            mass = Chem.Descriptors.ExactMolWt(prod_mol)

                            fp = AllChem.GetMorganFingerprint(prod_mol, 2)

                            # generate SVGs
                            if not isdir(SVG_DIRECTORY + '/' + result):
                                mkdir(SVG_DIRECTORY + '/' + result)
                            if not isdir(SVG_DIRECTORY + '/' + result + '/' + op):
                                mkdir(SVG_DIRECTORY + '/' + result + '/' + op)
                            img = create_img_no_highlights(prod_mol)
                            svg_name = SVG_DIRECTORY + '/' + result + '/' + op + '/' + prod[:-4] + 'svg'
                            with open(svg_name, 'w') as f:
                                f.write(img.data)

                            assert len(data['TemplateReaction']) == 1
                            reaction_id = data['TemplateReaction'][0]['ID']
                            assert len(reaction_id) == 1
                            reaction_id = reaction_id[0]
                            for rid in reaction_id:
                                reaction_ids.append(rid)

                            operators.append(op)
                            masses.append(mass)
                            # assert (mass >= low_lim) and (mass <= up_lim)
                            # prod_inchi = Chem.inchi.MolToInchi(prod_mol)
                            prod_inchikey = Chem.MolToInchiKey(prod_mol)
                            prod_inchikeys.append(prod_inchikey)
                            prod_inchikey = prod_inchikey.split("-")[0]
                            prod_inchikeys_block1.append(prod_inchikey)
                            prod_s = Chem.MolToSmiles(prod_mol)
                            prod_smiles.append(prod_s)


                            count += 1

        num_unique_operators = len(set(operators))
        num_prods = len(prod_inchikeys_block1)
        unique_prods = set(prod_inchikeys_block1)
        num_unique_prods = len(unique_prods)
        unique_ik = set(prod_inchikeys)
        occurences = []
        for prod in unique_prods:
            occurences.append(prod_inchikeys_block1.count(prod))

        stats.append(
            [result, pair['name'], pair['delta_mass'], target_mass, num_unique_operators, num_prods, num_unique_prods,
             prod_inchikeys, occurences, operators, masses, reaction_ids, prod_smiles])


        stats_numprods.append(count)

        #save to ranked output if there is only one candidate and no need for ranking
        if num_unique_prods == 1:
            if not os.path.isdir(WRITE_DIRECTORY + 'ranked_derivatives/' + result):
                mkdir(WRITE_DIRECTORY + 'ranked_derivatives/' + result)
            f = open(curr_filename)
            json_prod = json.load(f)
            new_filename = WRITE_DIRECTORY + 'ranked_derivatives/' + result + '/1_' + prod_inchikey + '.json'
            with open(new_filename, 'w') as f:
                json.dump(json_prod, f, indent=2)




    stats_numcorrect = np.array(stats_numcorrect)
    stats_numprods = np.array(stats_numprods)


    with open(WRITE_DIRECTORY + 'PROXIMAL2_results.csv', 'w') as f:
        write = csv.writer(f)
        col_names = ["PairId", "CompoundName", "DeltaMass", "TrueSuspectMass", "NumberOfUniqueOperators", "NumberOfProducts",
                        "NumberOfUniqueProducts", "UniqueIK", "NumberOfTimesEachIKProduced", "Operators", "Masses", "ReactionIDs",
                        "AllSmiles"]
        write.writerow(col_names)
        write.writerows(stats)

    df = pd.DataFrame(stats, columns=col_names)


    return df

def analyze_ground_truth_cands(WRITE_DIRECTORY, pairs):
    SVG_DIRECTORY = WRITE_DIRECTORY + 'svgs/'
    OUTPUT_DIRECTORY = WRITE_DIRECTORY + 'products/'
    if not isdir(SVG_DIRECTORY):
        mkdir(SVG_DIRECTORY)

    if not os.path.isdir(WRITE_DIRECTORY + 'ranked_derivatives/'):
        mkdir(WRITE_DIRECTORY + 'ranked_derivatives/')

    with open('../data/input/GMN_ids.csv', 'r') as f:
        ids = list(csv.reader(f, delimiter=","))
    annotated_inchi = {}
    annotated_mass = {}
    anno_clusterid = {}
    anno_names = {}
    anno_inchi_id = {}
    annotated_inchikey = {}
    for id in ids:
        anno_names[id[8]] = id[0]
        if id[5].strip() != '':
            anno_inchi_id[id[8]] = id[5]
            annotated_inchikey[id[0]] = id[11]

    ground_truth = {}
    groundtruth_masses = {}
    ground_truth_names = {}
    groundtruth_ops = {}
    for id, pair in pairs.iterrows():
        clusterid1 = pair['ID']
        clusterid2 = pair['suspect_ID']
        mass1 = pair['mass']
        mass2 = pair['suspect_mass']
        delta1 = pair['delta_mass']
        filename1 = clusterid1 + '_' + str(delta1)

        if filename1 not in ground_truth.keys():
            ground_truth[filename1] = [pair['suspect_inchikey']] #inchikey
        else:
            ground_truth[filename1].append(pair['suspect_inchikey'])
        groundtruth_masses[filename1] = mass2

        if filename1 not in ground_truth_names.keys():
            ground_truth_names[filename1] = [pair['suspect_name']]
        else:
            if pair[1] not in ground_truth_names[filename1]:
                ground_truth_names[filename1].append(pair['suspect_name'])

    prod_stats = []
    stats = []
    stats_numcorrect = []
    stats_numprods = []
    stats_tinchi = []
    stats_ginchi = []
    stats_tmol = []
    stats_wrongmass = []
    pcfp_dict = {}
    save_count = 0
    # iterate through the input instead
    for id, pair in tqdm.tqdm(pairs.iterrows(), desc="Finding Correct Matches", unit="pair"):

        result = pair['ID'] + '_' + str(pair['suspect_ID'])


        # set up vars
        true_inchikey = pair['suspect_inchikey']
        true_prodname = pair['suspect_name']

        target_mass = pair['suspect_mass']
        true_inchikey_block1 = true_inchikey.split("-")[0]
        true_mol = []
        count = 0
        correct = 0
        correct_tinchi = 0
        correct_tmol = 0
        masses = []
        correct_rpair = 0
        incorrect_rpair = 0
        prod_inchikeys = []
        prod_inchikeys_block1 = []
        prod_smiles = []
        operators = []
        reaction_ids = []
        similarities_to_true_suspect_ecfp = []
        similarities_to_true_suspect_maccs = []
        # similarity
        m1 = Chem.MolFromSmiles(pair['smiles'])
        fp1 = AllChem.GetMorganFingerprint(m1, 2)
        m2 = Chem.MolFromSmiles(pair['suspect_smiles'])
        fp2 = AllChem.GetMorganFingerprint(m2, 2)
        # maccs fp
        maccs_fp1 = MACCSkeys.GenMACCSKeys(m1)
        maccs_fp2 = MACCSkeys.GenMACCSKeys(m2)

        # calculate similarities
        true_sim_tanimoto = DataStructs.TanimotoSimilarity(fp1, fp2)
        largest_sim_tanimoto = None
        sim_tanimoto_prod = None

        true_sim_tanimoto_pc = None
        largest_sim_tanimoto_pc = None
        sim_tanimoto_prod_pc = None

        true_sim_tanimoto_maccs = DataStructs.TanimotoSimilarity(maccs_fp1, maccs_fp2)
        largest_sim_tanimoto_maccs = None
        sim_tanimoto_prod_maccs = None

        if os.path.isdir(OUTPUT_DIRECTORY + "/" + result):


            for op in listdir(OUTPUT_DIRECTORY + '/' + result):
                if op[0] != '.':
                    for prod in listdir(OUTPUT_DIRECTORY + '/' + result + '/' + op):
                        f = open(OUTPUT_DIRECTORY + '/' + result + '/' + op + '/' + prod)
                        data = json.load(f)
                        assert len(data['GeneratedProduct']) == 1
                        prod_j = data['GeneratedProduct'][0]
                        prod_mol = Chem.MolFromMolBlock(prod_j['mol'])
                        if prod_mol is not None:
                            curr_filename = OUTPUT_DIRECTORY + '/' + result + '/' + op + '/' + prod
                            mass = Chem.Descriptors.ExactMolWt(prod_mol)

                            fp = AllChem.GetMorganFingerprint(prod_mol, 2)
                            sim_tanimoto_ecfp = DataStructs.TanimotoSimilarity(fp2, fp)

                            if largest_sim_tanimoto == None or largest_sim_tanimoto < sim_tanimoto_ecfp:
                                largest_sim_tanimoto = sim_tanimoto_ecfp
                                sim_tanimoto_prod = op + '/' + prod

                            # sim of maccs fp
                            maccs_fp = MACCSkeys.GenMACCSKeys(prod_mol)
                            sim_tanimoto_maccs = DataStructs.TanimotoSimilarity(maccs_fp2, maccs_fp)
                            if largest_sim_tanimoto_maccs == None or largest_sim_tanimoto_maccs < sim_tanimoto_maccs:
                                largest_sim_tanimoto_maccs = sim_tanimoto_maccs
                                sim_tanimoto_prod_maccs = op + '/' + prod



                            # generate SVGs
                            if not isdir(SVG_DIRECTORY + '/' + result):
                                mkdir(SVG_DIRECTORY + '/' + result)
                            if not isdir(SVG_DIRECTORY + '/' + result + '/' + op):
                                mkdir(SVG_DIRECTORY + '/' + result + '/' + op)
                            img = create_img_no_highlights(prod_mol)
                            svg_name = SVG_DIRECTORY + '/' + result + '/' + op + '/' + prod[:-4] + 'svg'
                            with open(svg_name, 'w') as f:
                                f.write(img.data)

                            assert len(data['TemplateReaction']) == 1
                            reaction_id = data['TemplateReaction'][0]['ID']
                            assert len(reaction_id) == 1
                            reaction_id = reaction_id[0]
                            for rid in reaction_id:
                                reaction_ids.append(rid)

                            operators.append(op)
                            masses.append(mass)
                            # assert (mass >= low_lim) and (mass <= up_lim)
                            # prod_inchi = Chem.inchi.MolToInchi(prod_mol)
                            prod_inchikey = Chem.MolToInchiKey(prod_mol)
                            prod_inchikeys.append(prod_inchikey)
                            prod_inchikey = prod_inchikey.split("-")[0]
                            prod_inchikeys_block1.append(prod_inchikey)
                            prod_s = Chem.MolToSmiles(prod_mol)
                            prod_smiles.append(prod_s)

                            similarities_to_true_suspect_ecfp.append(sim_tanimoto_ecfp)
                            similarities_to_true_suspect_maccs.append(sim_tanimoto_maccs)

                            if prod_inchikey == true_inchikey_block1:
                                correct = 1
                                correct_tinchi = 1


                            count += 1

        num_unique_operators = len(set(operators))
        num_prods = len(prod_inchikeys_block1)
        unique_prods = set(prod_inchikeys_block1)
        num_unique_prods = len(unique_prods)
        unique_ik = set(prod_inchikeys)
        num_correct = prod_inchikeys_block1.count(true_inchikey_block1)
        occurences = []
        for prod in unique_prods:
            occurences.append(prod_inchikeys_block1.count(prod))

        stats.append(
            [result, pair['name'], true_prodname, pair['delta_mass'], target_mass, correct, num_correct,
             num_unique_operators,
             num_prods, num_unique_prods, true_inchikey,
             prod_inchikeys, occurences, operators, masses, reaction_ids,
             true_sim_tanimoto, sim_tanimoto_prod, largest_sim_tanimoto,
             true_sim_tanimoto_maccs, sim_tanimoto_prod_maccs, largest_sim_tanimoto_maccs, prod_smiles,
             similarities_to_true_suspect_ecfp, similarities_to_true_suspect_maccs])

        stats_numcorrect.append(correct)
        stats_numprods.append(count)
        stats_tmol.append(correct_tmol)
        stats_tinchi.append(correct_tinchi)

        # save to ranked output if there is only one candidate and no need for ranking
        if num_unique_prods == 1:
            if not os.path.isdir(WRITE_DIRECTORY + 'ranked_derivatives/' + result):
                mkdir(WRITE_DIRECTORY + 'ranked_derivatives/' + result)
            f = open(curr_filename)
            json_prod = json.load(f)
            new_filename = WRITE_DIRECTORY + 'ranked_derivatives/' + result + '/1_' + prod_inchikey + '.json'
            with open(new_filename, 'w') as f:
                json.dump(json_prod, f, indent=2)


    stats_numcorrect = np.array(stats_numcorrect)
    stats_numprods = np.array(stats_numprods)
    print("Number of Times a Correct Product was Generated: ", np.sum(stats_numcorrect))
    print("Average Number of Products Generated: ", np.mean(stats_numprods))
    print("Standard Deviation: ", np.std(stats_numprods))

    with open(WRITE_DIRECTORY + 'PROXIMAL2_results_validation.csv', 'w') as f:
        write = csv.writer(f)
        col_names = ["ClusterId", "CompoundName", "TrueNeighborName", "DeltaMass", "TrueSuspectMass", "Correct?",
                        "NumberOfTimesCorrectIKProduced", "NumberOfUniqueOperators", "NumberOfProducts",
                        "NumberOfUniqueProducts",
                        "TrueIK", "UniqueIK", "NumberOfTimesEachIKProduced", "Operators", "Masses", "ReactionIDs",
                        "TrueTanimotoSimilarityECFP", "ProductWithLargestTanimotoSimECFP", "LargestTanimotoSimECFP",
                        "TrueTanimotoSimilarityMACCS", "ProductWithLargestTanimotoSimMACCS", "LargestTanimotoSimMACCS",
                        "AllSmiles", "AllSimECFP", "AllSimMACCS"]
        write.writerow(col_names)
        write.writerows(stats)

    df = pd.DataFrame(stats, columns=col_names)


    return df


def process_ground_truth_cands_for_SOM(reactions, results, SOM_DIRECTORY, WRITE_DIRECTORY):
    ###############################################
    # Get all input necessary for all 3 datas of only the datapoints valid for all three
    # Most common operator, most generated product, and GNN-SOM
    ############################################################

    # get config enzyme info
    with open(SOM_DIRECTORY + '/data/config.json', 'r') as f:
        config = json.load(f)
    config['features']['enzyme'] = [tuple(ec) for ec in config['features']['enzyme']]

    col_names = list(results.columns)
    col_names.append('UniqueEnzymes')

    to_save = []
    for id, result in results.iterrows():


        # only if it is correct
        if result['Correct?'] == 1:
            # only if more than one product is generated
            if result['NumberOfUniqueProducts'] != 1:

                smiles = result['AllSmiles']
                target_mass = result['TrueSuspectMass']
                low_lim = float(target_mass) - 1.5
                up_lim = float(target_mass) + 1.5
                for s in smiles:
                    m = Chem.MolFromSmiles(s)
                    assert m is not None
                    if m is not None:
                        mass = Chem.Descriptors.ExactMolWt(m)
                        assert (mass >= low_lim) and (mass <= up_lim)

                operators = result['Operators']

                enzymes = []
                for op in operators:
                    idx = reactions.index[reactions['id'] == op].tolist()[0]
                    EC_list = reactions.iloc[idx]['EC']
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

                assert len(enzymes) == len(set(enzymes))

                if len(enzymes) > 0:
                    temp = result.to_list()
                    temp.append(enzymes)
                    to_save.append(temp)

    with open(WRITE_DIRECTORY + 'filtered_cands_for_ranking.csv', 'w') as f:
        write = csv.writer(f)
        write.writerow(col_names)
        write.writerows(to_save)

    return to_save

def process_cands_for_SOM(reactions, results, SOM_DIRECTORY, WRITE_DIRECTORY):

    # get config enzyme info
    with open(SOM_DIRECTORY + '/data/config.json', 'r') as f:
        config = json.load(f)
    config['features']['enzyme'] = [tuple(ec) for ec in config['features']['enzyme']]

    col_names = list(results.columns)
    col_names.append('UniqueEnzymes')

    to_save = []
    for id, result in results.iterrows():
        if result['NumberOfUniqueProducts'] > 1:

            smiles = result['AllSmiles']
            target_mass = result['TrueSuspectMass']
            low_lim = float(target_mass) - 1.5
            up_lim = float(target_mass) + 1.5
            for s in smiles:
                m = Chem.MolFromSmiles(s)
                assert m is not None
                if m is not None:
                    mass = Chem.Descriptors.ExactMolWt(m)
                    assert (mass >= low_lim) and (mass <= up_lim)

            operators = result['Operators']

            enzymes = []
            for op in operators:
                idx = reactions.index[reactions['id'] == op].tolist()[0]
                EC_list = reactions.iloc[idx]['EC']
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
            assert len(enzymes) == len(set(enzymes))

            if len(enzymes) > 0:
                temp = result.to_list()
                temp.append(enzymes)
                to_save.append(temp)

    with open(WRITE_DIRECTORY + 'filtered_cands_for_ranking.csv', 'w') as f:
        write = csv.writer(f)
        write.writerow(col_names)
        write.writerows(to_save)

    return to_save

def generate_kcf_ground_truth(WRITE_DIRECTORY, inputs, results):

    smiles_dict = {}
    for id, input in inputs.iterrows():
        if input['ID'] not in smiles_dict.keys():
            smiles_dict[input['ID']] = input['smiles']
        else:
            assert smiles_dict[input['ID']] == input['smiles']

    saved = []
    count = 0
    if not os.path.exists(WRITE_DIRECTORY + 'kcf'):
        os.mkdir(WRITE_DIRECTORY + 'kcf')
    for result in results:
        if result[5] == 1:
            if result[0].count('_') == 1:
                comps = result[0].split('_')
                comp1 = comps[0]
                comp2 = comps[1]

                file1 = WRITE_DIRECTORY + "kcf/" + comp1 + ".kcf"
                file2 = WRITE_DIRECTORY + "kcf/" + comp2 + ".kcf"

            if result[0].count('_') == 3:
                comps = result[0].split('_')
                comp1 = comps[0] + '_' + comps[1]
                comp2 = comps[2] + '_' + comps[3]
                file1 = WRITE_DIRECTORY + "kcf/" + comp1 + ".kcf"
                file2 = WRITE_DIRECTORY + "kcf/" + comp2 + ".kcf"

            if comp1 not in saved and not os.path.exists(file1):
                m = Chem.MolFromSmiles(smiles_dict[comp1])
                vec = KCFvec()
                vec.input_rdkmol(m, cpd_name=comp1)
                vec.convert_kcf_vec()
                f = open(file1, "w")
                f.write(vec.kcf)
                f.close()
                saved.append(comp1)
            if comp2 not in saved and not os.path.exists(file2):
                m = Chem.MolFromSmiles(smiles_dict[comp2])
                vec = KCFvec()
                vec.input_rdkmol(m, cpd_name=comp2)
                vec.convert_kcf_vec()
                f = open(file2, "w")
                f.write(vec.kcf)
                f.close()
                saved.append(comp2)

        count += 1

    return


def generate_kcf(WRITE_DIRECTORY, inputs, results):

    smiles_dict = {}
    for id, input in inputs.iterrows():
        if input['ID'] not in smiles_dict.keys():
            smiles_dict[input['ID']] = input['smiles']
        else:
            assert smiles_dict[input['ID']] == input['smiles']

    saved = []
    count = 0
    if not os.path.exists(WRITE_DIRECTORY + 'kcf'):
        os.mkdir(WRITE_DIRECTORY + 'kcf')
    for result in results:
        if result[0].count('_') == 1:
            comps = result[0].split('_')
            comp1 = comps[0]
            comp2 = comps[1]

            file1 = WRITE_DIRECTORY + "kcf/" + comp1 + ".kcf"
            file2 = WRITE_DIRECTORY + "kcf/" + comp2 + ".kcf"

        if result[0].count('_') == 3:
            comps = result[0].split('_')
            comp1 = comps[0] + '_' + comps[1]
            comp2 = comps[2] + '_' + comps[3]
            file1 = WRITE_DIRECTORY + "kcf/" + comp1 + ".kcf"
            file2 = WRITE_DIRECTORY + "kcf/" + comp2 + ".kcf"

        if comp1 not in saved and not os.path.exists(file1):
            m = Chem.MolFromSmiles(smiles_dict[comp1])
            vec = KCFvec()
            vec.input_rdkmol(m, cpd_name=comp1)
            vec.convert_kcf_vec()
            f = open(file1, "w")
            f.write(vec.kcf)
            f.close()
            saved.append(comp1)
        if comp2 not in saved and not os.path.exists(file2):
            m = Chem.MolFromSmiles(smiles_dict[comp2])
            vec = KCFvec()
            vec.input_rdkmol(m, cpd_name=comp2)
            vec.convert_kcf_vec()
            f = open(file2, "w")
            f.write(vec.kcf)
            f.close()
            saved.append(comp2)

        count += 1

    return