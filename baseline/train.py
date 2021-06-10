import os
from args import parse_args
from dkt.dataloader import Preprocess
from dkt import trainer
import torch
from dkt.utils import setSeeds
import wandb
import pickle
def main(args):
    wandb.login()
    
    setSeeds(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device

    preprocess = Preprocess(args)
    preprocess.load_train_data(args.file_name)
    train_data = preprocess.get_train_data()  # 전체 user(6698명) 별 testId, assessmentItemID, KnowledgeTag, answerCode
    
    train_data, valid_data = preprocess.split_data(train_data)  # 4688 2010(val ratio: 0.3)
    # with open('train_data.txt', 'rb') as f:
    #     train_data = pickle.load(f)  # 단 한줄씩 읽어옴
    # with open('valid_data.txt', 'rb') as f:
    #     valid_data = pickle.load(f)  # 단 한줄씩 읽어옴

    wandb.init(project='dkt', config=vars(args), tags=[args.model], name=f'custom_valid_{args.model}_{args.n_epochs}epochs')
    trainer.run(args, train_data, valid_data)
    

if __name__ == "__main__":
    args = parse_args(mode='train')
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)