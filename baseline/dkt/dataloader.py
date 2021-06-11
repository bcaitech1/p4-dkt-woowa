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
        # 라벨 인코딩할 columns
        cate_cols = ['assessmentItemID', 'testId', 'KnowledgeTag']
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

        # def convert_time(s):
        #     timestamp = time.mktime(datetime.strptime(s, '%Y-%m-%d %H:%M:%S').timetuple())
        #     return int(timestamp)
        #
        # df['time_stamp'] = df['Timestamp'].apply(convert_time)

        return df

    def __feature_engineering(self, df):
        df = df.sort_values(by=['userID', 'Timestamp'], axis=0)
        # feature 생성을 하지 않은 경우
        df = create_time_stamp(df)
        df = create_day(df)
        df = create_elapsed_time(df)
        df = create_lag_time(df)
        df = create_prior_acc(df)

        # 사용할 colums 선택
        use_features = ['userID',
                        'answerCode',  # 이 순서 유지
                        'assessmentItemID',
                        'testId',
                        'KnowledgeTag',
                        'elapsed_time',
                        'lag_time',
                        'prior_acc'
                        ]
        df = df[use_features]

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

        # df = df.sort_values(by=['userID','Timestamp'], axis=0)
        # columns = ['userID', 'assessmentItemID', 'testId', 'answerCode', 'KnowledgeTag']
        # group = df[columns].groupby('userID').apply(
        #         lambda r: (
        #             r['testId'].values,
        #             r['assessmentItemID'].values,
        #             r['KnowledgeTag'].values,
        #             r['answerCode'].values
        #         )
        #     )
        columns = df.columns  # userID가 맨 처음에 있어야함, 범주형 피처들이 앞쪽에 있어야함, 직접 지정하는게 나을듯
        group = df[columns].groupby('userID').apply(
            lambda r: tuple([r[col].values for col in columns[1:]])
        )
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
        # user 1명의 sequence data
        row = self.data[index]

        # 각 data의 sequence length
        seq_len = len(row[0])

        # load_data_from_file에서 사용한 columns의 순서대로
        correct, question, test, tag = row[0], row[1], row[2], row[3]
        cate_cols = [test, question, tag, correct]

        # max seq len을 고려하여서 이보다 길면 자르고 아닐 경우 그대로 냅둔다
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
            cont_cols = list(row[4:])
            if seq_len > self.args.max_seq_len:
                for i, col in enumerate(cont_cols):
                    cont_cols[i] = col[-self.args.max_seq_len:]
            for i, col in enumerate(cont_cols):
                cont_cols[i] = torch.tensor(col)

            return cont_cols + cate_cols

        return cate_cols

    def __len__(self):
        return len(self.data)


from torch.nn.utils.rnn import pad_sequence


def collate(batch):
    col_n = len(batch[0])
    col_list = [[] for _ in range(col_n)]
    # is_cont=True이면 DKTDataset의 output을 cont+cate로 해야 그대로 mask가 맨 마지막에 위치함
    max_seq_len = len(batch[0][-1])  # args.max_seq_len

    # batch의 값들을 각 column끼리 그룹화
    for row in batch:
        for i, col in enumerate(row):
            pre_padded = torch.zeros(max_seq_len)
            pre_padded[-len(col):] = col  # 앞부분이 0으로 padding됨
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


