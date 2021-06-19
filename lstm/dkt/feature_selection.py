# 1. 안쓸거면 []로 선언해주기
# 2. CATEGORICAL: pre, post 둘다 쓸거면 pre 먼저

DEFAULT = ['userID', 'answerCode', 'assessmentItemID', 'testId', 'KnowledgeTag']
CATEGORICAL = ['testPre', 'testPost']
CONTINUOUS = ['user_correct_answer', 'user_total_answer', 'user_acc',
'test_mean', 'test_sum', 'tag_mean', 'tag_sum', 'item_ans_rate',
'time_stamp', 'elapsed_time', 'lag_time', 'user_recent_acc_5',
'user_recent_acc_10', 'user_recent_acc_30', 'user_recent_acc_50', 'user_recent_acc_100', 
'rel_point', 'sum_rel_point', 'knowledgetag_stroke',
'test_preId_mean', 'test_preId_sum', 'test_postId_mean', 'test_postId_sum',
'test_time', 'test_time_mean', 'test_time_sum']#, 'day']