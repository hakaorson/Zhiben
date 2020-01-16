'''
用于返回处理好的数据
'''
from Preprocess import readfile
from Preprocess import feature
from Preprocess import split


class final_struct():
    def __init__(self, item):
        self.dgl_data = item[0]
        self.mol_feat = item[1]
        self.target = item[2]


def data_zipper(*itors):
    items = zip(*itors)
    result = [final_struct(item) for item in items]
    return result


def main(path):
    data_from_file = readfile.data_reader(path)
    data_from_feature = feature.feature_extractor(data_from_file.all_smiles)
    data_ziped = data_zipper(data_from_feature.all_dgl,
                             data_from_feature.all_mol_feat, data_from_file.all_targets)
    data_from_split = split.splitor(data_ziped)
    return data_from_split


if __name__ == '__main__':
    main('Data/bbbp_test.csv')
