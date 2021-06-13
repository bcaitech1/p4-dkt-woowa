import os
from numpy.core.numerictypes import _construct_lookups
import torch
import numpy as np
from torch._C import device

from .dataloader import get_loaders
from .optimizer import get_optimizer
from .scheduler import get_scheduler
from .criterion import get_criterion
from .metric import get_metric
from .model import LSTM, LSTMATTN, Bert, Saint, LastQuery

import wandb
import time
import datetime

from .feature_selection import *


# Get current learning rate
# def get_lr(scheduler):
#     return scheduler.get_last_lr()[0]

# for plateua schduler
def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']


def run(args, train_data, valid_data):
    print(f'<< run: {args.is_cont} >>')
    train_loader, valid_loader = get_loaders(args, train_data, valid_data)

    # only when using warmup scheduler
    args.total_steps = int(len(train_loader.dataset) / args.batch_size) * (args.n_epochs)
    args.warmup_steps = args.total_steps // 10

    model = get_model(args)
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)

    best_auc = -1
    early_stopping_counter = 0
    for epoch in range(args.n_epochs):

        print(f"Start Training: Epoch {epoch + 1}")
        start = time.time()
        ### TRAIN
        train_auc, train_acc, train_loss = train(train_loader, model, optimizer, args)

        ### VALID
        auc, acc, _, _ = validate(valid_loader, model, args)

        sec = time.time() - start
        times = str(datetime.timedelta(seconds=sec)).split(".")
        times = times[0]
        print(f'<<<<<<<<<<  {epoch + 1} EPOCH spent : {times}  >>>>>>>>>>')

        # model save or early stopping
        wandb.log({"epoch": epoch, "train_loss": train_loss, "train_auc": train_auc, "train_acc": train_acc,
                   "valid_auc": auc, "valid_acc": acc, "Learning_rate": get_lr(optimizer), })
        if auc > best_auc:
            best_auc = auc
            # torch.nn.DataParallel로 감싸진 경우 원래의 model을 가져옵니다.
            model_to_save = model.module if hasattr(model, 'module') else model
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_to_save.state_dict(),
            },
                args.model_dir, 'model.pt',
            )
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= args.patience:
                print(f'EarlyStopping counter: {early_stopping_counter} out of {args.patience}')
                break

        # scheduler
        if args.scheduler == 'plateau':
            scheduler.step(best_auc)
        else:
            scheduler.step()


def train(train_loader, model, optimizer, args):
    print(f'<< train: {args.is_cont} >>')
    model.train()

    total_preds = []
    total_targets = []
    losses = []
    for step, batch in enumerate(train_loader):
        input = process_batch(batch, args)
        preds = model(input)
        targets = input[0]  # correct

        loss = compute_loss(preds, targets)
        update_params(loss, model, optimizer, args)

        if step % args.log_steps == 0:
            print(f"Training steps: {step} Loss: {str(loss.item())}")

        # predictions
        preds = preds[:, -1]
        targets = targets[:, -1]

        if args.device == 'cuda':
            preds = preds.to('cpu').detach().numpy()
            targets = targets.to('cpu').detach().numpy()
        else:  # cpu
            preds = preds.detach().numpy()
            targets = targets.detach().numpy()

        total_preds.append(preds)
        total_targets.append(targets)
        losses.append(loss)

    total_preds = np.concatenate(total_preds)
    total_targets = np.concatenate(total_targets)

    # Train AUC / ACC
    auc, acc = get_metric(total_targets, total_preds)
    loss_avg = sum(losses) / len(losses)
    print(f'TRAIN AUC : {auc} ACC : {acc}')
    return auc, acc, loss_avg


def validate(valid_loader, model, args):
    print(f'<< validate: {args.is_cont} >>')
    model.eval()

    total_preds = []
    total_targets = []
    for step, batch in enumerate(valid_loader):
        input = process_batch(batch, args)

        preds = model(input)
        targets = input[0]  # correct

        # predictions
        preds = preds[:, -1]
        targets = targets[:, -1]

        if args.device == 'cuda':
            preds = preds.to('cpu').detach().numpy()
            targets = targets.to('cpu').detach().numpy()
        else:  # cpu
            preds = preds.detach().numpy()
            targets = targets.detach().numpy()

        total_preds.append(preds)
        total_targets.append(targets)

    total_preds = np.concatenate(total_preds)
    total_targets = np.concatenate(total_targets)

    # Train AUC / ACC
    auc, acc = get_metric(total_targets, total_preds)

    print(f'VALID AUC : {auc} ACC : {acc}\n')

    return auc, acc, total_preds, total_targets


def inference(args, test_data):
    print(f'<< inference: {args.is_cont} >>')
    model = load_model(args)
    model.eval()
    _, test_loader = get_loaders(args, None, test_data)

    total_preds = []

    for step, batch in enumerate(test_loader):
        input = process_batch(batch, args)

        preds = model(input)

        # predictions
        preds = preds[:, -1]

        if args.device == 'cuda':
            preds = preds.to('cpu').detach().numpy()
        else:  # cpu
            preds = preds.detach().numpy()

        total_preds += list(preds)

    write_path = os.path.join(args.output_dir, "output.csv")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(write_path, 'w', encoding='utf8') as w:
        print("writing prediction : {}".format(write_path))
        w.write("id,prediction\n")
        for id, p in enumerate(total_preds):
            w.write('{},{}\n'.format(id, p))


def get_model(args):
    print(f"<< get_model : {args.is_cont} >>")
    """
    Load model and move tensors to a given devices.
    """
    #cont_count = 3 # dummy?
    cont_count = len(CONTINUOUS)
    
    if args.model == 'lstm': model = LSTM(args)
    if args.model == 'lstmattn': model = LSTMATTN(args)
    if args.model == 'bert': model = Bert(args)
    if args.model == 'saint': model = Saint(args)
    if args.model == 'lastquery': model = LastQuery(args)

    model.to(args.device)

    return model


# 배치 전처리
def process_batch(batch, args):
    # print(f"<< process_batch : {args.is_cont} >>")
    # 범주형 피처 컬럼 5개
    #test, question, tag, correct, mask = batch[-5:]  # [batch_size(64), max_seq_len(20)]
    categorical_features = list(batch[-(len(DEFAULT)+len(CATEGORICAL)):])

    # change to float
    #mask = mask.type(torch.FloatTensor)
    #correct = correct.type(torch.FloatTensor)
    mask = categorical_features[-1].type(torch.FloatTensor) # categorical 중 마지막
    correct = categorical_features[0].type(torch.FloatTensor) # categorical 중 처음

    # interaction: 과거 정답 여부를 다음 시퀀스에 추가적인 feature로 사용하게끔 한칸 시프트 해준 feature
    #  interaction을 임시적으로 correct를 한칸 우측으로 이동한 것으로 사용
    #    saint의 경우 decoder에 들어가는 input이다
    interaction = correct + 1  # 패딩을 위해 correct값에 1을 더해준다. (정답 2, 오답 1)
    interaction = interaction.roll(shifts=1, dims=1)
    interaction_mask = mask.roll(shifts=1, dims=1)
    interaction_mask[:, 0] = 0
    interaction = (interaction * interaction_mask).to(torch.int64)  # 가장 마지막으로 푼 문제를 제외하고 정답 2, 오답 1
    # print(interaction)
    # exit()
    #  test_id, question_id, tag
    '''
    test = ((test + 1) * mask).to(torch.int64)
    question = ((question + 1) * mask).to(torch.int64)
    tag = ((tag + 1) * mask).to(torch.int64)
    '''
    for i in range(1, len(categorical_features)-1, 1): # 0번째인 correct, 마지막인 mask는 제외
        categorical_features[i] = ((categorical_features[i] + 1)*mask).to(torch.int64)


    if args.is_cont:
        #cont = batch[:-5]
        cont = batch[:-len(categorical_features)] # 수정
        # cont features도 padding을 위해 1을 더함
        for i in range(len(cont)):
            cont[i] = ((cont[i] + 1) * mask).to(torch.float32)

    # device memory로 이동
    '''
    test = test.to(args.device)
    question = question.to(args.device)
    tag = tag.to(args.device)
    correct = correct.to(args.device)
    mask = mask.to(args.device)
    interaction = interaction.to(args.device)
    '''
    categorical_features = [correct.to(args.device)] + [_.to(args.device) for _ in categorical_features[1:-1]] + [mask.to(args.device), interaction.to(args.device)]

    if args.is_cont:
        for i in range(len(cont)):
            cont[i] = cont[i].to(args.device)
        '''
        return (test, question,
                tag, correct, mask,
                interaction, cont)
        '''
        categorical_features.append(cont)
        return tuple(categorical_features)
    '''
    return (test, question,
            tag, correct, mask,
            interaction)
    '''
    return tuple(categorical_features)

# loss계산하고 parameter update!
def compute_loss(preds, targets):
    """
    Args :
        preds   : (batch_size, max_seq_len)
        targets : (batch_size, max_seq_len)

    """
    loss = get_criterion(preds, targets)
    # 마지막 시퀀드에 대한 값만 loss 계산
    loss = loss[:, -1]
    loss = torch.mean(loss)
    return loss


def update_params(loss, model, optimizer, args):
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
    optimizer.step()
    optimizer.zero_grad()


def save_checkpoint(state, model_dir, model_filename):
    print('saving model ...')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(state, os.path.join(model_dir, model_filename))


def load_model(args):
    model_path = os.path.join(args.model_dir, args.model_name)
    print("Loading Model from:", model_path)
    load_state = torch.load(model_path)
    model = get_model(args)

    # 1. load model state
    model.load_state_dict(load_state['state_dict'], strict=True)

    print("Loading Model from:", model_path, "...Finished.")
    return model