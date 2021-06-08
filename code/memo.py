def create_elapeed_lag_time(df):
    elapsed_time_list = []
    lag_time_list = []

    # 유저별
    for user_id in df.userID.unique():
        lag_time = 0
        end_time = 0
        user_df = df[df.userID == user_id]
        test_ids_df = user_df.groupby('testId')
        for testid in user_df.testId.unique():
            time_stamps = test_ids_df.get_group(testid).time_stamp.values
            if end_time:
                lag_time = time_stamps[0] - end_time
            for i in range(len(time_stamps)):
                lag_time_list.append(lag_time)
                if i == 0:
                    elapsed_time_list.append(0)
                else:
                    elapsed_time_list.append(time_stamps[i] - time_stamps[i - 1])
                    end_time = time_stamps[i]

    return elapsed_time_list, lag_time_list


def create_solved(df):
    solved = []
    for user_id in df.userID.unique():
        user_df = df[df.userID == user_id]
        key_list = list(set(df[df.userID == user_id].KnowledgeTag.values))
        solved_dict = dict(zip(key_list, [0]*len(key_list)))
        for i in range(len(user_df)):
            row = user_df.iloc[i]
            if not solved_dict[row.KnowledgeTag]:
                solved.append(0)
                solved_dict[row.KnowledgeTag] = 1
            else:
                solved.append(1)
    return solved