import numpy as np

# 设置随机数种子
np.random.seed(1)


# 按照8：1：1划分train/valid/test
def split_train_valid_test(feature, label):
    shuffled_indices = np.random.permutation(len(feature))
    train_start, valid_start, test_start = 0, int(len(feature) * 0.8), int(len(feature) * 0.9)
    train_indices, valid_indices = shuffled_indices[train_start: valid_start], shuffled_indices[valid_start: test_start]
    test_indices = shuffled_indices[test_start:]

    return {
        "train": {
            "features": feature[train_indices],
            "labels": label[train_indices]
        },
        "valid": {
            "features": feature[valid_indices],
            "labels": label[valid_indices]
        },
        "test": {
            "features": feature[test_indices],
            "labels": label[test_indices]
        }
    }
