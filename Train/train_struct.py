from Train import split
import torch
import random
from sklearn import metrics


class batch_generator():
    def __init__(self, data, batch_size):
        self.data = data
        random.shuffle(self.data)
        self.batch_size = batch_size
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        try:
            self.data[self.index+self.batch_size-1]  # 用于检查是否越界
            b_data = self.data[self.index:self.index+self.batch_size]
        except IndexError:
            raise StopIteration()
        self.index += self.batch_size
        return b_data


def simple(args, model: torch.nn.Module, data):
    data_split = split.splitor(args, data)
    loss_fnc = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(args.epoch):
        data_generator = batch_generator(
            data_split.train_data, args.batch_size)
        for batch in data_generator:
            loss = 0
            for item in batch:
                target = torch.tensor(item.target, dtype=torch.float32)
                logits = model(item.dgl_data, item.mol_feat)
                loss += loss_fnc(logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    test(model, data_split.test_data)


def test(model, data):
    predict = []
    target = []
    for item in data:
        proba = model(item.dgl_data, item.mol_feat).detach().numpy()
        pred = 0 if proba < 0.5 else 1
        predict.append(pred)
        target.append(int(item.target[0]))
    auc = metrics.roc_auc_score(target, predict)
    print(auc)


if __name__ == '__main__':
    pass
