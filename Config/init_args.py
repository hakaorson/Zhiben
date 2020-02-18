import argparse


def init_train_pars():
    parser = argparse.ArgumentParser(description='init')
    # 系统相关
    parser.add_argument('--gpu', type=int, default=-1, help="gpu")
    # 数据相关
    parser.add_argument('--data_path', type=str, default='Data/bbbp.csv')
    parser.add_argument('--split', type=str, default='8_1_1')
    # 模型相关参数
    parser.add_argument('--hidden_feat_size', type=int, default=10)
    parser.add_argument('--atom_feat_size', type=int, default=-1)
    parser.add_argument('--bond_feat_size', type=int, default=-1)
    parser.add_argument('--mol_feat_size', type=int, default=-1)
    parser.add_argument('--hidden_layer_num', type=int, default=3)
    # 训练相关参数
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--epoch', type=int, default=10)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    pass
