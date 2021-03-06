from torch import nn
import torch
import dgl
import random
random.seed(666)
torch.random.manual_seed(666)


def get_toy_data():
    dgl_graph = dgl.DGLGraph()
    dgl_graph.add_nodes(1, {'n_input': torch.tensor(
        [1, 1, 1, 1], dtype=torch.float32).reshape(1, -1)})
    dgl_graph.add_nodes(1, {'n_input': torch.tensor(
        [2, 2, 2, 2], dtype=torch.float32).reshape(1, -1)})
    dgl_graph.add_nodes(1, {'n_input': torch.tensor(
        [3, 3, 3, 3], dtype=torch.float32).reshape(1, -1)})
    dgl_graph.add_nodes(1, {'n_input': torch.tensor(
        [4, 4, 4, 4], dtype=torch.float32).reshape(1, -1)})
    dgl_graph.add_edge(0, 1, {'e_input': torch.tensor(
        [0.1, 0.1, 0.1], dtype=torch.float32).reshape(1, -1)})
    dgl_graph.add_edge(1, 2, {'e_input': torch.tensor(
        [1.2, 1.2, 1.2], dtype=torch.float32).reshape(1, -1)})
    dgl_graph.add_edge(0, 2, {'e_input': torch.tensor(
        [0.2, 0.2, 0.2], dtype=torch.float32).reshape(1, -1)})
    dgl_graph.add_edge(2, 3, {'e_input': torch.tensor(
        [2.3, 2.3, 2.3], dtype=torch.float32).reshape(1, -1)})

    dgl_graph.add_edge(1, 0, {'e_input': torch.tensor(
        [0.1, 0.1, 0.1], dtype=torch.float32).reshape(1, -1)})
    dgl_graph.add_edge(2, 1, {'e_input': torch.tensor(
        [1.2, 1.2, 1.2], dtype=torch.float32).reshape(1, -1)})
    dgl_graph.add_edge(2, 0, {'e_input': torch.tensor(
        [0.2, 0.2, 0.2], dtype=torch.float32).reshape(1, -1)})
    dgl_graph.add_edge(3, 2, {'e_input': torch.tensor(
        [2.3, 2.3, 2.3], dtype=torch.float32).reshape(1, -1)})
    return dgl_graph


class FullConn(nn.Module):
    def __init__(self, mol_feat_size, dgl_feat_size):
        super().__init__()
        self.weight_full = nn.Parameter(
            torch.rand((mol_feat_size+dgl_feat_size, 1), requires_grad=True))

    def forward(self, feat):
        result = torch.squeeze(torch.mm(feat, self.weight_full))
        return result


class FeatFusion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, dgl_feat, mol_feat):
        return torch.cat((dgl_feat, mol_feat), -1).reshape(1, -1)


class ReadOut(nn.Module):
    def __init__(self, input_atom_feat_size, hidden_feat_size):
        super().__init__()
        self.weight_output = nn.Parameter(
            torch.rand((input_atom_feat_size+hidden_feat_size, hidden_feat_size), requires_grad=True))

    def readout_msg(self, edge):
        edge_data = edge.data['hidden']
        return {'edge_mail': edge_data}

    def readout_reduce(self, node):
        node_mail = torch.sum(node.mailbox['edge_mail'], 1)
        return {'out_sum': node_mail}

    def readout_nodeupdate(self, node):
        node_origin_feat = node.data['n_input']
        node_sum_feat = node.data['out_sum']
        node_feat_cat = torch.cat((node_origin_feat, node_sum_feat), -1)
        node_feat_mm = torch.mm(node_feat_cat, self.weight_output)
        # TODO 激活函数
        node_feat_final = node_feat_mm
        return {'feat_final': node_feat_final}

    def forward(self, dgl_data: dgl.DGLGraph):
        dgl_data.update_all(
            self.readout_msg, self.readout_reduce, self.readout_nodeupdate)
        gcn_feat = dgl_data.ndata['feat_final']
        result = torch.sum(gcn_feat, 0)
        return result


class FeatConvert(nn.Module):
    def __init__(self, input_atom_feat_size, input_edge_feat_size, hidden_feat_size):
        super().__init__()
        self.weight_feat2hidden_atom = nn.Parameter(
            torch.rand((input_atom_feat_size, hidden_feat_size), requires_grad=True))
        self.weight_feat2hidden_edge = nn.Parameter(
            torch.rand((input_edge_feat_size, hidden_feat_size), requires_grad=True))

    def forward(self, dgl_data: dgl.DGLGraph):
        node_feature = dgl_data.ndata['n_input']
        node_feature_new = torch.mm(node_feature, self.weight_feat2hidden_atom)
        edge_feature = dgl_data.edata['e_input']
        edge_feature_new = torch.mm(edge_feature, self.weight_feat2hidden_edge)
        # TODO 添加激活函数
        dgl_data.ndata['n_input'] = node_feature_new
        dgl_data.edata['e_input'] = edge_feature_new
        return dgl_data


class EdgeFeatInit(nn.Module):
    def __init__(self, input_atom_feat_size, input_edge_feat_size, hidden_feat_size):
        super().__init__()
        self.weight_feat2hidden = nn.Parameter(
            torch.rand((input_atom_feat_size+input_edge_feat_size, hidden_feat_size), requires_grad=True))

    def edge_init(self, edge):
        edge_data = edge.data['e_input']
        edge_src = edge.src['n_input']
        edge_concate = torch.cat((edge_data, edge_src), -1)
        edge_init = torch.mm(edge_concate, self.weight_feat2hidden)
        # TODO 添加激活函数
        return {'init': edge_init, 'hidden': edge_init}

    def forward(self, dgl_data: dgl.DGLGraph):
        dgl_data.apply_edges(self.edge_init)
        return dgl_data


class SingleLayer(nn.Module):
    def __init__(self, hidden_feat_size):
        super().__init__()
        self.weight_h2h_edge = nn.Parameter(torch.rand(
            hidden_feat_size, hidden_feat_size), requires_grad=True)

    def gcn_msg(self, edge):  # 结点到边的信息传递（通过src和dst获取源点和目标点的特征）
        # 将所有的边，每一个维度的数据整合在一起，比如100条边，每条边有长度为8的向量，那么转换后就成了10*100的张量
        edge_data = edge.data['hidden']
        return {'edge_mail': edge_data}

    def gcn_reduce(self, node):  # 边到结点的汇聚（通过mailbox函数获取结点的邻边的所有信息）
        # 选取n个结点*feature，选择多少个结点是临时决定的（由计算量决定）
        node_mail = torch.sum(node.mailbox['edge_mail'], 1)  # 代表把边进行汇合
        return {'mail': node_mail}

    def edge_update(self, edge):
        pseud_converge = edge.src['mail']-edge.data['hidden']
        feature_mm = torch.mm(pseud_converge, self.weight_h2h_edge)
        feature_add_init = torch.add(feature_mm, edge.data['init'])
        # TODO 激活函数
        feature_output = feature_add_init
        return {'hidden': feature_output}

    def forward(self, dgl_data: dgl.DGLGraph):
        # 注意这个函数指挥更新node feature
        dgl_data.update_all(self.gcn_msg, self.gcn_reduce)
        dgl_data.apply_edges(self.edge_update)
        return dgl_data


class DMPNN(nn.Module):
    def __init__(self, hidden_feat_size, hidden_layer_num):
        super().__init__()
        self.hidden_layers = nn.ModuleList()
        for index in range(hidden_layer_num):
            self.hidden_layers.append(SingleLayer(hidden_feat_size))

    def forward(self, dgl_data):
        for layer in self.hidden_layers:
            dgl_data = layer(dgl_data)
        return dgl_data


class MainModel(nn.Module):
    def __init__(self, input_atom_feat_size, input_edge_feat_size, hidden_feat_size, hidden_layer_num, mol_feat_size):
        super().__init__()
        self.convert = FeatConvert(
            input_atom_feat_size, input_edge_feat_size, hidden_feat_size)
        self.edge_init = EdgeFeatInit(
            input_atom_feat_size, input_edge_feat_size, hidden_feat_size)
        self.dmpnn = DMPNN(hidden_feat_size, hidden_layer_num)
        self.graph_reader = ReadOut(input_atom_feat_size, hidden_feat_size)
        self.feat_fusion = FeatFusion()
        self.full_conn = FullConn(mol_feat_size, hidden_feat_size)

    def forward(self, dgl_data, mol_feat):
        '''
        dgl_after_conv = self.convert(dgl_data)
        dgl_after_gcn = self.dmpnn(dgl_after_conv)
        '''
        dgl_after_init = self.edge_init(dgl_data)
        dgl_after_gcn = self.dmpnn(dgl_after_init)
        dgl_feat = self.graph_reader(dgl_after_gcn)
        fusion_feat = self.feat_fusion(dgl_feat, mol_feat)
        result = self.full_conn(fusion_feat)
        return result


def train():
    model = MainModel(4, 3, 2, 5, 3)
    toy_dgl = get_toy_data()
    model.train()
    logits = model(toy_dgl, torch.tensor([8, 8, 8], dtype=torch.float32))
    return logits


if __name__ == '__main__':
    train()
