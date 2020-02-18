from Preprocess import preprocess
from Model import basemodel
from Config import init_args
from Train import train_struct

if __name__ == '__main__':
    args = init_args.init_train_pars()
    dataset = preprocess.data_zip_all(args)
    model = basemodel.MainModel(args)
    train_struct.train(args, model, dataset)
    pass
