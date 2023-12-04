from rdkit import Chem
from rdkit.Chem import Draw
from IPython.display import SVG

import pubchempy
from pubchempy import get_compounds, Compound

import os

import re
import json

import sys

import importlib
PROXIMAL2 = importlib.import_module("../PROXIMAL2-main")
from PROXIMAL2.proximal_functions.GenerateMolFiles2 import ConstructMol

def create_img_no_highlights(mol):
    dwg = Draw.MolDraw2DSVG(450, 300)
    dwg.DrawMolecule(mol)
    dwg.FinishDrawing()
    img = SVG(dwg.GetDrawingText().replace('svg:', ''))
    return img

def get_compound_mass_dict(filename):
    if filename is None:
        with open('../../data/kegg/compound/compound') as f:
            lines = f.readlines()
    else:
        with open(filename+'/compound') as f:
            lines = f.readlines()

    compound_mass = {}
    c_count = 0
    line_count = 0
    while line_count < len(lines):
        assert lines[line_count][0:12] == "ENTRY       "
        c_name = lines[line_count][12:].split("Compound")[0].strip()
        line_count += 1
        while lines[line_count][0:12] != "EXACT_MASS  " and lines[line_count][0:3] != "///":
            line_count += 1
        if lines[line_count][0:12] == "EXACT_MASS  ":
            mass = lines[line_count][12:].strip()
            assert c_name not in compound_mass.keys()
            compound_mass[c_name] = mass
            c_count += 1

        while lines[line_count][0:3] != "///":
            line_count += 1
        line_count += 1

    return compound_mass

def calculate_delta_mass_operator(pair, compound_mass, filename):
    if filename is None:
        directory_name = '../../data/kegg/mol/'
    else:
        directory_name = filename+'/mol/'

    masses = []
    kegg_masses = []
    for compound in pair:
        if os.path.isfile(directory_name + compound + ".mol"):
            m = Chem.MolFromMolFile(directory_name + compound + ".mol")
            i = Chem.MolToInchi(m)
            s = Chem.MolToSmiles(m)

            try:
                c = get_compounds(i, 'inchi')

            except:
                c = None
            if c is None:
                try:
                    c = get_compounds(s, 'smiles')

                except pubchempy.BadRequestError:
                    c = None
            if c is not None and len(c) > 0:
                assert len(c) == 1
                c = c[0]
                mass = c.exact_mass
            else:
                mass = None

        else:
            mass = None

        if compound in compound_mass.keys():
            kegg_mass = compound_mass[compound]
            kegg_masses.append(kegg_mass)
            if mass == None:
                mass = kegg_mass

        masses.append(mass)

    note = []
    if None in masses:
        if len(kegg_masses) == 2:
            delta = abs((float(kegg_masses[0]) - float(kegg_masses[1])).__round__(4))
        else:
            delta = None
    else:
        delta = abs((float(masses[0]) - float(masses[1])).__round__(4))
        if len(kegg_masses) == 2:
            delta_kegg = abs((float(kegg_masses[0]) - float(kegg_masses[1])).__round__(4))
            if round(float(delta)) != round(float(delta_kegg)):
                note = [masses, delta, kegg_masses, delta_kegg]

    return delta, note

def select_operators_by_mass(mass, margin, mass_keys, delta_masses):
    if (mass_keys[-1] < (abs(mass) + margin)):
        lower=len(mass_keys)+1
    else:
        lower = next(x for x, val in enumerate(mass_keys) if val >= (abs(mass) - margin))

    if (mass_keys[-1] <= (abs(mass) + margin)):
        upper = len(mass_keys)
    else:
        upper = next(x for x, val in enumerate(mass_keys) if val > (abs(mass) + margin))
    dataRP = set()
    for i in range(lower, upper):
        RPs = delta_masses[mass_keys[i]]
        for rp in RPs:
            dataRP.add(rp)

    return list(dataRP)

def add_delta_mass(reactions, metabolites):
    # with open("FinalMetabolicSpace_prova_nodeltamass.pkl", 'rb') as f:
    #     reactions = pickle.load(f)

    # metabolites = pd.read_csv("metabolites_retrorules_all.csv")

    # idx = reactions.index[reactions['deltaMass'] == ['na']]#.tolist()

    for index, row in reactions.iterrows():
        if row['deltaMass'] == ['na']:
            pair = row['Pair']
            assert len(pair) == 2
            mass1 = metabolites.loc[metabolites['id'] == pair[0]]['mass'].item()
            mass2 = metabolites.loc[metabolites['id'] == pair[1]]['mass'].item()
            row['deltaMass'] = [mass2 - mass1]
        else:
            row['deltaMass'] = [float(row['deltaMass'][0])]


    return reactions

#Common
def ExtractPairs_BAM(reactions, metabolites):
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    import pandas as pd

    MetabolicSpace = pd.DataFrame(columns=['id', 'rxn', 'Pair', 'EC', 'deltaMass'])
    n_rxn = 0
    for idx in reactions.index:
        print("Removing Redundancy among Pairs. Reaction: %d/%d" % (idx, len(reactions) - 1))
        substrates = reactions.namesFormula[idx].split(' -> ')[0].strip().split(' + ')
        products = reactions.namesFormula[idx].split(' -> ')[1].strip().split(' + ')
        compoundPair = []
        if len(substrates) > 1 and len(products) > 1:
            # Version to avoid association when all the substrate and products have a low similarity
            mol_sub = {}
            for s in substrates:
                a = Chem.MolFromSmiles(metabolites.loc[metabolites['name'].isin( \
                    [s]), 'smiles'].values[0])
                mol_sub[s] = a
            mol_prod = {}
            for p in products:
                a = Chem.MolFromSmiles(metabolites.loc[metabolites['name'].isin( \
                    [p]), 'smiles'].values[0])
                mol_prod[p] = a

            for s_key, s_value in mol_sub.items():
                f_a = AllChem.GetMorganFingerprint(s_value, 2)
                sim = {}
                for p_key, p_value in mol_prod.items():
                    tmp_fp = AllChem.GetMorganFingerprint(p_value, 2)
                    sim[p_key] = DataStructs.DiceSimilarity(f_a, tmp_fp)
                    if sim[p_key] < 0.5:
                        sim.pop(p_key)

                if all([x == 0 for x in sim.values()]):
                    continue
                max_sim = max([x for x in sim.values()])
                compoundPair.append([s_key, [x for x, y in sim.items() if y == max_sim][0]])
        else:
            for sub in substrates:
                for prod in products:
                    compoundPair.append([sub, prod])

        for each_p in compoundPair:
            tmp = MetabolicSpace.loc[MetabolicSpace.Pair.isin([each_p]),]
            if len(tmp) > 0:
                MetabolicSpace.loc[tmp.index[0], 'rxn'] += [reactions.id[idx]]

                try:
                    tmp_ec = re.findall("[\d-]+\.[\d-]+\.[\d-]+\.[\d-]+", reactions.EC[idx])
                    tmp_ec = MetabolicSpace.loc[tmp.index[0], 'EC'] + tmp_ec
                except TypeError:
                    tmp_ec = MetabolicSpace.loc[tmp.index[0], 'EC']
                MetabolicSpace.loc[tmp.index[0], 'EC'] = list(set(tmp_ec))
            else:
                MetabolicSpace.loc[n_rxn, 'id'] = "R%d" % (n_rxn)
                MetabolicSpace.loc[n_rxn, 'rxn'] = [reactions.id[idx]]
                MetabolicSpace.loc[n_rxn, 'Pair'] = each_p
                MetabolicSpace.loc[n_rxn, 'deltaMass'] = [reactions.deltaMass[idx]]
                try:
                    tmp_ec = re.findall("[\d-]+\.[\d-]+\.[\d-]+\.[\d-]+", reactions.EC[idx])
                except TypeError:
                    tmp_ec = []
                MetabolicSpace.loc[n_rxn, 'EC'] = tmp_ec
                n_rxn += 1
    return MetabolicSpace

#GenerateMolFiles
def SaveInputFiles_BAM(inputList, inputStructure, inputListNumbering, charge, filename):

    try:
        smiles = Chem.MolToSmiles(Chem.MolFromMolBlock(ConstructMol(inputStructure,
                                                        charge, {}, "", "", "", "")))
        mol = Chem.MolToMolBlock(Chem.MolFromSmiles(smiles))
    except:
        smiles = ""

        mol = ConstructMol(inputStructure, charge, {}, "", "", "", "")

    file_to_save = {'inputList': inputList, 'inputStructure': inputStructure, 'inputListNumbering': inputListNumbering,
                    'charge': charge, 'mol': mol}

    with open(filename, 'w') as f:
        json.dump(file_to_save, f, indent=2)

    return


def ConstructMolFile_BAM(inputStructure, charge, charge_Template, filename, product, query_info):
    # Instead of saving the mol just as text, generate a json with a more complete information within it
    # Store the information as InChI
    try:
        smiles = Chem.MolToSmiles(Chem.MolFromMolBlock(ConstructMol(inputStructure,
                                                                    charge, charge_Template, query_info['ID'],
                                                                    product['ID'][0], product['KCF']['compound1'],
                                                                    product['KCF']['compound2'])))
        mol = Chem.MolToMolBlock(Chem.MolFromSmiles(smiles))
    except:
        smiles = ""
        mol = ConstructMol(inputStructure, charge, charge_Template, query_info['ID'], product['ID'][0],
                           product['KCF']['compound1'], product['KCF']['compound2'])

    ec = product['Enzyme']
    rxnID = product['Reaction']
    TemplateSubstrate = product['KCF']['compound1']
    TemplateProduct = product['KCF']['compound2']
    QueryName = query_info['name']
    QuerySmiles = query_info['smiles']
    QueryID = query_info['ID']
    ObtainedFromInput = product['ObtainedFromInput']
    file_to_save = {'GeneratedProduct': [
        {'smiles': smiles,
         'mol': mol}
    ],
        'TemplateReaction': [
            {'ec': ec,
             'ID': rxnID,
             'Substrate': TemplateSubstrate,
             'Product': TemplateProduct}
        ],
        'QueryInformation': [
            {'name': QueryName,
             'ID': QueryID,
             'smiles': QuerySmiles}
        ],
        'ObtainedFromInput': ObtainedFromInput
    }
    with open(filename, 'w') as f:
        json.dump(file_to_save, f, indent=2)


