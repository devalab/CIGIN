import warnings
from collections import OrderedDict
from copy import deepcopy
import numpy as np


from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors as rdDesc
from rdkit.Chem import AllChem
from rdkit import rdBase
from rdkit import RDLogger

import torch

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)
rdBase.DisableLog('rdApp.error')
warnings.filterwarnings("ignore")

device = "cuda" if torch.cuda.is_available() else "cpu"

def one_of_k_encoding(x, allowable_set):
    '''Function to get one hot encoding'''
    
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    '''Maps inputs not in the allowable set to the last element.'''
    
    if x not in allowable_set:
        x = allowable_set[-1]
        
    return list(map(lambda s: x == s, allowable_set))


def bond_features(bond, use_chirality=True,bond_length=None):
    '''Bond level features from rdkit bond object'''

    bt = bond.GetBondType()
    bond_feats = [
      bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
      bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
      bond.GetIsConjugated(),
      bond.IsInRing()
    ]
    if bond_length is not None:
        bond_feats = bond_feats + [bond_length]
    if use_chirality:
        bond_feats = bond_feats + one_of_k_encoding_unk(
            str(bond.GetStereo()),
            ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])
    return np.array(bond_feats)

def atom_features(atom,stereo,features,bool_id_feat=False,explicit_H=False):
    '''Atom level features from rdkit's atom object '''
    if bool_id_feat:
        return np.array([atom_to_id(atom)])
    else:
        results = one_of_k_encoding_unk(
          atom.GetSymbol(),
          [
            'C',
            'N',
            'O',
            'S',
            'F',
            'P',
            'Cl',
            'Br',
            'I',
            'Si'
          ]) + one_of_k_encoding(atom.GetDegree(),
                                 [0, 1, 2, 3, 4]) + \
                  one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1]) + \
                  one_of_k_encoding_unk(atom.GetNumRadicalElectrons(), [0, 1]) + \
                  one_of_k_encoding_unk(atom.GetFormalCharge(), [-1, 0, 1]) + \
                  one_of_k_encoding_unk(atom.GetHybridization(), [
                    Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                    Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D]) + \
                  [int(i) for i in list("{0:06b}".format(features))]

    if not explicit_H:
          results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                [0, 1, 2, 3, 4])
    
    try:
        results = results + one_of_k_encoding_unk(
            stereo,
            ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
    except Exception as e:
        results = results + [False, False
                          ] + [atom.HasProp('_ChiralityPossible')]
        
    return np.array(results)

def ConstructMolecularGraph(molecule):
    
    '''Constructs molecular graph from rdkit's molecule object '''
    
    g = OrderedDict({})
    h = OrderedDict({})
    
    molecule = Chem.MolFromSmiles(molecule)
    stereo = Chem.FindMolChiralCenters(molecule)
    features = rdDesc.GetFeatureInvariants(molecule)
    chiral_centers = [0]* molecule.GetNumAtoms()
    for i in stereo:
        chiral_centers[i[0]] = i[1]
    for i in range(0, molecule.GetNumAtoms()):
        atom_i = molecule.GetAtomWithIdx(i)
        h[i] = torch.FloatTensor(atom_features(atom_i,chiral_centers[i],features[i]).astype(np.float64)).to(device)
        for j in range(0, molecule.GetNumAtoms()):
            e_ij = molecule.GetBondBetweenAtoms(i, j)

            if e_ij != None:
                e_ij =  map(lambda x: 1 if x == True else 0, bond_features(e_ij).tolist()) # ADDED edge feat
                e_ij = torch.FloatTensor(list(e_ij)).to(device)
                atom_j = molecule.GetAtomWithIdx(j)
                if i not in g:
                    g[i] = []
                g[i].append( (e_ij, j) )

    return g, h