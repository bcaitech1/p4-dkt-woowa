
import time
from datetime import datetime

def convert_time(s):
    timestamp = time.mktime(datetime.strptime(s, '%Y-%m-%d %H:%M:%S').timetuple())
    return int(timestamp)

def create_time_stamp(df):
    df['time_stamp'] = df['Timestamp'].apply(convert_time)
    return df

def create_day(df):
    df['day'] = df['Timestamp'].apply(lambda x: x.split()[0])
    return df

def create_elapsed_time(df):
    prev_timestamp = df.groupby(['userID', 'testId', 'day'])[['time_stamp']].shift()
    df['elapsed_time'] = df['time_stamp'] - prev_timestamp['time_stamp']
    df['elapsed_time'] = df['elapsed_time'].fillna(0)
    return df


def create_lag_time(df):
    start_end_id_by_user_test = df.groupby(['userID', 'testId', 'day']).apply(
        lambda x: (x.index.values[0], x.index.values[-1])).reset_index()
    start_end_id_by_user_test = start_end_id_by_user_test.sort_values(by=[0]).reset_index(drop=True)
    start_row_id_by_user = start_end_id_by_user_test.groupby('userID').apply(lambda x: x.index.values[0])

    lag_time_list = [0] * len(df)
    for start_row, end_row in start_end_id_by_user_test[0][1:]:
        start_time = df.time_stamp[start_row]
        prev_time = df.time_stamp[start_row - 1]
        lag_time = start_time - prev_time
        lag_time_list[start_row:end_row + 1] = [lag_time] * (end_row - start_row + 1)

    # 사용자가 바뀌는 부분 첫 시험지 lag_time은 0으로 변경
    for user_start_idx in start_row_id_by_user:
        start, end = start_end_id_by_user_test.loc[user_start_idx][0]
        lag_time_list[start:end + 1] = [0] * (end - start + 1)

    df['lag_time'] = lag_time_list

    return df


def create_prior_acc(df):
    df['prior_acc_count'] = df.groupby('userID')['answerCode'].transform(lambda x: x.cumsum().shift(1)).fillna(0)
    # 이전까지 푼 문제 수  (피처 계산에만 사용)
    df['prior_quest_count'] = df.groupby('userID')['answerCode'].cumcount().fillna(0)
    # 이전 문제까지의 정답률
    df['prior_acc'] = (df['prior_acc_count'] / df['prior_quest_count']).fillna(0)
    return df