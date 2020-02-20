'''
用于产生feature
包括graph_feature,node_feature,edge_feature
'''
from rdkit import Chem
import rdkit
import dgl
import torch
from descriptastorus.descriptors import rdNormalizedDescriptors
import numpy as np

# Atom feature sizes
MAX_ATOMIC_NUM = 100
ATOM_FEATURES = {
    'atomic_num': list(range(MAX_ATOMIC_NUM)),
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-1, -2, 1, 2, 0],
    'chiral_tag': [0, 1, 2, 3],
    'num_Hs': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],
}
BOND_FEATURES = {'stereo': [0, 1, 2, 3, 4, 5]}


class feature_extractor():
    def __init__(self, args, smiles_data):
        self.set_args(args, smiles_data[0])
        self.feature_data = [feature_data_item(item) for item in smiles_data]
        self.all_dgl = [item.dgl_data for item in self.feature_data]
        self.all_mol_feat = [item.mol_feat for item in self.feature_data]

    def set_args(self, args, smiles):  # 根据样本获取特征的规模
        feated_smiles = feature_data_item(smiles)
        nsize = feated_smiles.dgl_data.ndata['n_input'].size()[-1]
        esize = feated_smiles.dgl_data.edata['e_input'].size()[-1]
        molsize = feated_smiles.mol_feat.size()[-1]
        args.atom_feat_size = nsize
        args.bond_feat_size = esize
        args.mol_feat_size = molsize


class feature_data_item():
    def __init__(self, smiles):
        self.smiles = smiles
        self.mol = Chem.MolFromSmiles(self.smiles)
        self.dgl_data = self.get_dgl_graph(self.mol)
        self.mol_feat = self.get_mol_feature(self.smiles)

    def onehot_embed(self, val, choice):
        encoding = [0] * (len(choice) + 1)
        index = choice.index(val) if val in choice else -1
        encoding[index] = 1
        return encoding

    def get_dgl_graph(self, mol):
        dgl_graph = dgl.DGLGraph()
        atoms_num = mol.GetNumAtoms()
        for atom in mol.GetAtoms():
            node_feature = torch.tensor(
                self.get_node_feature(atom), dtype=torch.float32).reshape(1, -1)  # 保持维度
            dgl_graph.add_nodes(1, {'n_input': node_feature})
        for n_id_1 in range(atoms_num):
            for n_id_2 in range(n_id_1+1, atoms_num):
                bond = mol.GetBondBetweenAtoms(n_id_1, n_id_2)
                if bond is None:
                    continue
                bond_feature = torch.tensor(
                    self.get_bond_feature(bond), dtype=torch.float32).reshape(1, -1)  # 保持维度
                dgl_graph.add_edge(n_id_1, n_id_2, data={
                                   'e_input': bond_feature})
                dgl_graph.add_edge(n_id_2, n_id_1, data={
                                   'e_input': bond_feature})
        return dgl_graph

    def get_node_feature(self, mol_node):
        node_feature = []
        node_feature.extend(self.onehot_embed(
            int(mol_node.GetAtomicNum()) - 1, ATOM_FEATURES['atomic_num']))
        node_feature.extend(self.onehot_embed(
            int(mol_node.GetTotalDegree()), ATOM_FEATURES['degree']))
        node_feature.extend(self.onehot_embed(
            int(mol_node.GetFormalCharge()), ATOM_FEATURES['formal_charge']))
        node_feature.extend(self.onehot_embed(
            int(mol_node.GetChiralTag()), ATOM_FEATURES['chiral_tag']))
        node_feature.extend(self.onehot_embed(
            int(mol_node.GetTotalNumHs()), ATOM_FEATURES['num_Hs']))
        node_feature.extend(self.onehot_embed(
            int(mol_node.GetHybridization()), ATOM_FEATURES['hybridization']))
        node_feature.append(1 if mol_node.GetIsAromatic() else 0)
        node_feature.append(mol_node.GetMass() * 0.01)
        return node_feature

    def get_bond_feature(self, mol_bond):
        bond_feature = []
        bond_type = mol_bond.GetBondType()
        bond_feature.append(int(bond_type == Chem.rdchem.BondType.SINGLE))
        bond_feature.append(int(bond_type == Chem.rdchem.BondType.DOUBLE))
        bond_feature.append(int(bond_type == Chem.rdchem.BondType.TRIPLE))
        bond_feature.append(int(bond_type == Chem.rdchem.BondType.AROMATIC))
        bond_feature.append(int(mol_bond.GetIsConjugated())
                            if bond_type is not None else 0)
        bond_feature.append(int(mol_bond.IsInRing())
                            if bond_type is not None else 0)
        bond_feature.extend(self.onehot_embed(
            int(mol_bond.GetStereo()), BOND_FEATURES['stereo']))
        return bond_feature

    def get_mol_feature(self, smiles):
        '''
        使用morgan feature
        '''
        mol = Chem.MolFromSmiles(smiles)
        features_vec = Chem.AllChem.GetMorganFingerprintAsBitVect(
            mol, 2, nBits=2048)
        features = np.zeros((1,))
        rdkit.DataStructs.ConvertToNumpyArray(features_vec, features)
        features = torch.tensor(features, dtype=torch.float32)
        return features
        '''
        mol_feat_generator = rdNormalizedDescriptors.RDKit2DNormalized()
        mol_feature = mol_feat_generator.process(smiles)[1:]
        mol_feature = torch.tensor(mol_feature, dtype=torch.float32)
        return mol_feature
        '''


if __name__ == '__main__':
    pass
