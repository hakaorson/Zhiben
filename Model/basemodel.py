from torch import nn
import torch
import dgl


class FullConn(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        return dgl_data


class FeatFusion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        return dgl_data


class ReadOut(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        return dgl_data


class FeatConvert(nn.Module):
    def __init__(self, input_atom_feat_size, input_bond_feat_size, hidden_feat_size):
        super().__init__()
        self.weight_feat2hidden_atom = nn.Parameter(
            torch.rand((input_atom_feat_size, hidden_feat_size), requires_grad=True))
        self.weight_feat2hidden_bond = nn.Parameter(
            torch.rand((input_bond_feat_size, hidden_feat_size), requires_grad=True))

    def forward(self, dgl_data: dgl.DGLGraph):
        node_feature = dgl_data.ndata['n']
        node_feature_new = torch.mm(node_feature, self.weight_feat2hidden_atom)
        dgl_data.ndata['n'] = node_feature_new
        return dgl_data


class SingleLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        return dgl_data


class DMPNN(nn.Module):
    def __init__(self, hidden_feat_size, hidden_layer_num):
        super().__init__()
        self.hidden_layers = nn.ModuleList()
        for index in range(hidden_layer_num):
            self.hidden_layers.append(SingleLayer())

    def forward(self, dgl_data):
        return dgl_data


class MainModel(nn.Module):
    def __init__(self, input_atom_feat_size, input_bond_feat_size, hidden_feat_size, hidden_layer_num):
        super().__init__()
        self.convert = FeatConvert(
            input_atom_feat_size, input_bond_feat_size, hidden_feat_size)
        self.dmpnn = DMPNN(hidden_feat_size, hidden_layer_num)
        self.graph_reader = ReadOut()
        self.feat_fusion = FeatFusion()
        self.full_conn = FullConn()

    def forward(self, dgl_data, mol_feat):
        dgl_after_conv = self.convert(dgl_data)
        dgl_after_gcn = self.dmpnn(dgl_after_conv)
        dgl_feat = self.graph_reader(dgl_after_gcn)
        fusion_feat = self.fusion(dgl_feat, mol_feat)
        result = self.full_conn(fusion_feat)
        return result


def train(data):
    model = MainModel(8, 3, 10, 2)
    for item in data:
        target = item.target
        model.train()
        logits = model(item.dgl_data, item.mol_feat)
        pass
    pass


if __name__ == '__main__':
    pass
