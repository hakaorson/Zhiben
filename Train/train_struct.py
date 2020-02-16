def train(args, data):
    data_split = split.splitor(args, data)
    model = MainModel(args)
    loss_fnc = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters())
    for item in data:
        target = torch.tensor(item.target, dtype=torch.float32)
        model.train()
        logits = model(item.dgl_data, item.mol_feat)
        loss = loss_fnc(logits, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    pass
