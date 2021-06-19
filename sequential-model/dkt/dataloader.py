import os
from datetime import datetime
import time
import tqdm
import pandas as pd
import random
from sklearn.preprocessing import LabelEncoder
import numpy as np
import torch

from .features import *
from dkt.feature_selection import *


class Preprocess:
    def __init__(self, args):
        self.args = args
        self.train_data = None
        self.test_data = None

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

    def split_data(self, data, test_size=0.1, shuffle=True, seed=0):
        """
        split data into two parts with a given ratio.
        """
        if shuffle:
            random.seed(seed)  # fix to default seed 0
            random.shuffle(data)

        size = int(len(data) * (1 - test_size))
        data_1 = data[:size]
        data_2 = data[size:]

        return data_1, data_2

    def __save_labels(self, encoder, name):
        le_path = os.path.join(self.args.asset_dir, name + '_classes.npy')
        np.save(le_path, encoder.classes_)

    def __preprocessing(self, df, is_train=True):
        print(f'<< __preprocessing: {self.args.is_cont} >>')
        # 라벨 인코딩할 columns (userID, answerCode 제외)
        cate_cols = DEFAULT[2:] + CATEGORICAL

        # 로그 스케일 적용할 columns
        log1p_cols = ['elapsed_time',
                      'lag_time'
                      ]
        if self.args.is_cont:
            df[log1p_cols] = np.log1p(df[log1p_cols])

        if not os.path.exists(self.args.asset_dir):
            os.makedirs(self.args.asset_dir)

        for col in cate_cols:
            le = LabelEncoder()
            if is_train:
                # For UNKNOWN class
                a = df[col].unique().tolist() + ['unknown']
                le.fit(a)
                self.__save_labels(le, col)
            else:
                label_path = os.path.join(self.args.asset_dir, col + '_classes.npy')
                le.classes_ = np.load(label_path)

                df[col] = df[col].apply(lambda x: x if x in le.classes_ else 'unknown')

            # 모든 컬럼이 범주형이라고 가정
            df[col] = df[col].astype(str)
            test = le.transform(df[col])
            df[col] = test

        return df

    def __feature_engineering(self, df):
        df = df.sort_values(by=['userID', 'Timestamp'], axis=0)
        df = df[DEFAULT + CATEGORICAL + CONTINUOUS]  #사용할 colums 선택
        return df

    def load_data_from_file(self, file_name, is_train=True):
        print(f'<< load_data_from_file: {self.args.is_cont} >>')
        csv_file_path = os.path.join(self.args.data_dir, file_name)
        df = pd.read_csv(csv_file_path)  # 전체 피처가 있는 csv 파일
        df = self.__feature_engineering(df)  # 사용할 피처선택
        df = self.__preprocessing(df, is_train)  # 범주형 데이터 라벨 인코딩, 수치형 데이터 로그 스케일링

        # 추후 feature를 embedding할 시에 embedding_layer의 input 크기를 결정할때 사용
        self.args.n_questions = len(np.load(os.path.join(self.args.asset_dir, 'assessmentItemID_classes.npy')))
        self.args.n_test = len(np.load(os.path.join(self.args.asset_dir, 'testId_classes.npy')))
        self.args.n_tag = len(np.load(os.path.join(self.args.asset_dir, 'KnowledgeTag_classes.npy')))

        front_cols = ['userID', 'answerCode', 'assessmentItemID', 'testId', 'KnowledgeTag']
        for item in ('testPre', 'testPost', 'lag_time'):
            if item in df.columns:  # catetorical로 testPre나 testPost가 있을 경우
                front_cols.append(item)
                if item == 'testPre':
                    self.args.n_testpre = len(np.load(os.path.join(self.args.asset_dir, 'testPre_classes.npy')))
                elif item == 'testPost':
                    self.args.n_testpost = len(np.load(os.path.join(self.args.asset_dir, 'testPost_classes.npy')))
                elif item == 'lag_time':
                    self.args.n_lagtime = len(np.load(os.path.join(self.args.asset_dir, 'lag_time_classes.npy')))

        # column 순서 조정 (userID, answerCode가 맨앞, catetorical이 먼저 오도록)
        columns = [_ for _ in df.columns if _ not in (front_cols)]  # categorical을 일단 제외
        columns = front_cols + columns  # categorical을 맨앞으로
        print(f'features: {columns}')
        group = df[columns].groupby('userID').apply(
            lambda r: tuple([r[col].values for col in columns[1:]])
        )
        if is_train:
            return group
        return group.values

    def load_train_data(self, file_name):
        self.train_data = self.load_data_from_file(file_name)

    def load_test_data(self, file_name):
        self.test_data = self.load_data_from_file(file_name, is_train=False)


class DKTDataset(torch.utils.data.Dataset):
    def __init__(self, data, args):
        print(f'<< DKTDataset: {args.is_cont} >>')
        self.data = data
        self.args = args

    def __getitem__(self, index):
        row = self.data[index]
        seq_len = len(row[0])

        n_cate_cols = len(DEFAULT) + len(CATEGORICAL)
        cate_cols = [row[i] for i in range(n_cate_cols - 1)]  # n_cate_cols에서 userID 카운트 제거

        # max seq len을 고려하여서 이보다 길면 자름
        if seq_len > self.args.max_seq_len:
            for i, col in enumerate(cate_cols):
                cate_cols[i] = col[-self.args.max_seq_len:]
            mask = np.ones(self.args.max_seq_len, dtype=np.int16)
        else:
            mask = np.zeros(self.args.max_seq_len, dtype=np.int16)
            mask[-seq_len:] = 1

        # mask도 columns 목록에 포함시킴
        cate_cols.append(mask)

        # np.array -> torch.tensor 형변환
        for i, col in enumerate(cate_cols):
            cate_cols[i] = torch.tensor(col)

        if self.args.is_cont:
            # 범주형 데이터 이후
            cont_cols = list(row[n_cate_cols - 1:])
            if seq_len > self.args.max_seq_len:
                for i, col in enumerate(cont_cols):
                    cont_cols[i] = col[-self.args.max_seq_len:]
            for i, col in enumerate(cont_cols):
                try:
                    cont_cols[i] = torch.tensor(col)
                except:
                    print(i)
                    print(cont_cols[i])

            return cont_cols + cate_cols

        return cate_cols

    def __len__(self):
        return len(self.data)


def collate(batch):
    col_n = len(batch[0])
    col_list = [[] for _ in range(col_n)]
    max_seq_len = len(batch[0][-1])

    # batch의 값들을 각 column끼리 그룹화
    for row in batch:
        for i, col in enumerate(row):
            pre_padded = torch.zeros(max_seq_len)
            pre_padded[-len(col):] = col  # 앞부분 0으로 padding
            col_list[i].append(pre_padded)

    for i, _ in enumerate(col_list):
        col_list[i] = torch.stack(col_list[i])

    return tuple(col_list)


def get_loaders(args, train, valid):
    print(f'<< get_loaders: {args.is_cont} >>')

    pin_memory = True
    train_loader, valid_loader = None, None

    if train is not None:
        trainset = DKTDataset(train, args)
        train_loader = torch.utils.data.DataLoader(trainset, num_workers=args.num_workers, shuffle=True,
                                                   batch_size=args.batch_size, pin_memory=pin_memory,
                                                   collate_fn=collate)
    if valid is not None:
        valset = DKTDataset(valid, args)
        valid_loader = torch.utils.data.DataLoader(valset, num_workers=args.num_workers, shuffle=False,
                                                   batch_size=args.batch_size, pin_memory=pin_memory,
                                                   collate_fn=collate)

    return train_loader, valid_loader


def slidding_window(data, args):
    window_size = args.max_seq_len
    stride = args.stride

    augmented_datas = []
    for row in data:
        seq_len = len(row[0])

        # 만약 window 크기보다 seq len이 같거나 작으면 augmentation을 하지 않음
        if seq_len <= window_size:
            augmented_datas.append(row)
        else:
            total_window = ((seq_len - window_size) // stride) + 1

            # 앞에서부터 slidding window 적용
            for window_i in range(total_window):
                # window로 잘린 데이터를 모으는 리스트
                window_data = []
                for col in row:
                    window_data.append(col[window_i * stride:window_i * stride + window_size])

                # Shuffle
                # 마지막 데이터의 경우 shuffle을 하지 않음
                if args.shuffle and window_i + 1 != total_window:
                    shuffle_datas = shuffle(window_data, window_size, args)
                    augmented_datas += shuffle_datas
                else:
                    augmented_datas.append(tuple(window_data))

            # slidding window에서 뒷부분이 누락될 경우 추가
            total_len = window_size + (stride * (total_window - 1))
            if seq_len != total_len:
                window_data = []
                for col in row:
                    window_data.append(col[-window_size:])
                augmented_datas.append(tuple(window_data))

    return augmented_datas


def shuffle(data, data_size, args):
    shuffle_datas = []
    for i in range(args.shuffle_n):
        shuffle_data = []
        random_index = np.random.permutation(data_size)
        for col in data:
            shuffle_data.append(col[random_index])
        shuffle_datas.append(tuple(shuffle_data))
    return shuffle_datas


def data_augmentation(data, args):
    if args.window == True:
        data = slidding_window(data, args)

    return data