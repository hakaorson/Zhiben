'''
用于产生feature
包括graph_feature,node_feature,edge_feature
'''
from rdkit import Chem
import dgl
import torch
from descriptastorus.descriptors import rdNormalizedDescriptors


def onehot_embed(val, choice):
    return []


class feature_extractor():
    def __init__(self, smiles_data):
        self.feature_data = [feature_data_item(item) for item in smiles_data]
        self.all_dgl = [item.dgl_data for item in self.feature_data]
        self.all_mol_feat = [item.mol_feat for item in self.feature_data]


class feature_data_item():
    def __init__(self, smiles):
        self.smiles = smiles
        self.mol = Chem.MolFromSmiles(self.smiles)
        self.node_feat_size = 8  # TODO
        self.bond_feat_size = 3  # TODO
        self.mol_feat_size = 200  # TODO
        self.dgl_data = self.get_dgl_graph(self.mol)
        self.mol_feat = self.get_mol_feature(self.smiles)

    def get_dgl_graph(self, mol):
        dgl_graph = dgl.DGLGraph()
        atoms_num = mol.GetNumAtoms()
        for atom in mol.GetAtoms():
            node_feature = torch.tensor(
                self.get_node_feature(atom)).reshape(1, -1)  # 保持维度
            dgl_graph.add_nodes(1, {'n': node_feature})
        for n_id_1 in range(atoms_num):
            for n_id_2 in range(n_id_1+1, atoms_num):
                bond = mol.GetBondBetweenAtoms(n_id_1, n_id_2)
                if bond is None:
                    continue
                bond_feature = torch.tensor(
                    self.get_bond_feature(bond)).reshape(1, -1)  # 保持维度
                dgl_graph.add_edge(n_id_1, n_id_2, data={
                                   'e': bond_feature})
                dgl_graph.add_edge(n_id_2, n_id_1, data={
                                   'e': bond_feature})
        return dgl_graph

    def get_node_feature(self, mol_node):
        node_feature = []
        node_feature.append(int(mol_node.GetAtomicNum()))
        node_feature.append(int(mol_node.GetTotalDegree()))
        node_feature.append(int(mol_node.GetFormalCharge()))
        node_feature.append(int(mol_node.GetChiralTag()))
        node_feature.append(int(mol_node.GetTotalNumHs()))
        node_feature.append(int(mol_node.GetHybridization()))
        node_feature.append(int(mol_node.GetIsAromatic()))
        node_feature.append(float(mol_node.GetMass()))
        assert self.node_feat_size == len(node_feature)  # 保持长度一致
        return node_feature

    def get_bond_feature(self, mol_bond):
        bond_feature = []
        bond_feature.append(int(mol_bond.GetIsConjugated()))
        bond_feature.append(int(mol_bond.IsInRing()))
        bond_feature.append(int(mol_bond.GetStereo()))
        assert self.bond_feat_size == len(bond_feature)  # 保持长度一致
        return bond_feature

    def get_mol_feature(self, smiles):
        mol_feat_generator = rdNormalizedDescriptors.RDKit2DNormalized()
        mol_feature = mol_feat_generator.process(smiles)[1:]
        assert self.mol_feat_size == len(mol_feature)
        return mol_feature


if __name__ == '__main__':
    pass
