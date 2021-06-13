import os
from args import parse_args
from dkt.dataloader import Preprocess
from dkt import trainer
import torch
from dkt.utils import setSeeds
import wandb

import pickle
from prettyprinter import cpprint
from sklearn.model_selection import StratifiedKFold



def main(args):
    wandb.login()

    setSeeds(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device

    preprocess = Preprocess(args)
    preprocess.load_train_data(args.file_name)
    train_data = preprocess.get_train_data()  # 전체 user(6698명) 별 testId, assessmentItemID, KnowledgeTag, answerCode

    #train_data, valid_data = preprocess.split_data(train_data, test_size=args.test_size)  # 4688 2010(val ratio: 0.3)
    # cv7(유저별 정답률 skfold_0, seed 42)
    # with open('train_data.txt', 'rb') as f:
    #     train_data = pickle.load(f)
    # with open('valid_data.txt', 'rb') as f:
    #     valid_data = pickle.load(f)

    wandb.init(project='dkt', config=vars(args), tags=[args.model], name=f'{args.run_name}')

    #trainer.run(args, train_data, valid_data)

    # skfold
    with open('cv7_label_list.txt', 'rb') as f:
        label_list = pickle.load(f)

    with open('cv7_group_values_list.txt', 'rb') as f:
        group_values_list = pickle.load(f)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    skf.get_n_splits(group_values_list, label_list)
    fold = 1
    acc_list = []
    auc_list = []

    for train_index, valid_index in skf.split(group_values_list, label_list):
        print(f'-- fold {fold}')
        print("VALID:", valid_index[:5])
        train_data, valid_data = group_values_list[train_index], group_values_list[valid_index]
        #acc, auc = trainer.fold_run(args, train_data, valid_data, fold)
        trainer.run(args, train_data, valid_data)
        # acc_list.append(acc)
        # auc_list.append(auc)
        # print('')
        # print(f'Fold {fold} ACC: {acc}')
        # print(f'Fold {fold} AUC: {auc}')
        # print('')
        fold += 1
        break
    # # acc 평균
    # print(f'acc = {acc_list}')
    # print(f'auc = {auc_list}')
    # acc_mean = sum(acc_list) / len(acc_list)
    # auc_mean = sum(auc_list) / len(auc_list)
    # print(f'acc_mean: {acc_mean}')
    # print(f'auc_mean: {auc_mean}')

if __name__ == "__main__":
    args = parse_args(mode='train')
    os.makedirs(args.model_dir, exist_ok=True)
    cpprint(args)
    main(args)