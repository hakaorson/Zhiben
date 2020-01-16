'''
读取文件，返回数据list
样本格式为smiles, 属性1，属性2...
'''
import csv


class data_reader():
    def __init__(self, path):
        self.file_path = path
        self.origin_data = self.readfile(self.file_path)
        self.struct_data = [struct_data_item(item)
                            for item in self.origin_data]
        self.all_smiles = [item.smiles for item in self.struct_data]
        self.all_targets = [item.targets for item in self.struct_data]
        self.first_target = [item.targets[0] for item in self.struct_data]

    def readfile(self, path):
        with open(path) as file:
            reader = csv.reader(file)
            next(reader)
            origin_data = list(reader)
        return origin_data


class struct_data_item():
    def __init__(self, item):
        self.smiles = item[0]
        self.targets = list(map(float, item[1:]))


if __name__ == '__main__':
    pass
