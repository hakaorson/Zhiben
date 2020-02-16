from Preprocess import preprocess
from Model import basemodel
from Config import init_args

if __name__ == '__main__':
    args = init_args.init_train_pars()
    dataset = preprocess.main(args)
    basemodel.train(args, dataset)
    pass
