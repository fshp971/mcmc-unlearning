import numpy as np


def mia_get_threshold(train_conf, test_conf):
    result = []
    for val in train_conf: result.append( [val, 1] )
    for val in test_conf:  result.append( [val, 0] )

    train_cnt = len(train_conf)
    test_cnt  = len(test_conf)

    result = np.array(result, dtype=np.float32)
    result = result[result[:,0].argsort()]
    one = train_cnt
    zero = test_cnt
    best_atk_acc = -1
    threshold = -1 
    for i in range(len(result)):
        atk_acc = 0.5 * (one/train_cnt + (test_cnt-zero)/test_cnt)
        if best_atk_acc < atk_acc and threshold < result[i][0]:
            best_atk_acc = atk_acc
            threshold = result[i][0]
        if result[i][1] == 1:
            one = one-1
        else: zero = zero-1

    return threshold, best_atk_acc
