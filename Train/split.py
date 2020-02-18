'''
划分数据集
'''


class splitor():
    def __init__(self, args, data):
        self.data = data
        self.length = len(self.data)
        split_rate = self.split_rate(args.split)
        cut1 = int(split_rate[0]*self.length)
        cut2 = int((split_rate[0]+split_rate[1])*self.length)
        self.train_data = self.data[:cut1]
        self.valid_data = self.data[cut1:cut2]
        self.test_data = self.data[cut2:]

    def split_rate(self, string):
        nums = list(map(int, string.split('_')))
        sums = sum(nums)
        result = [item/sums for item in nums]
        return result


if __name__ == '__main__':
    pass
