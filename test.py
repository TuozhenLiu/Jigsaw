# -*- coding:utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Author: Tuozhen
# Date: 2021/12/9
# Description:
# ----------------------------------------------------------------------------------------------------------------------
class args:
    def __init__(self):
        self.train_path = '../input/jigsaw-toxic-severity-rating/validation_data.csv'
        self.modified_train_path = '../input/jigsawmodify-train-data/modified_train.csv'
        self.valid_path = '../input/jigsaw-toxic-severity-rating/validation_data.csv'
        self.eval_model = []
        self.model_path = '../input/jigsaw-toxic-severity-rating/validation_data.csv'
        self.checkpoint_path = '../input/jigsaw-toxic-severity-rating/validation_data.csv'
        self.pos_neg_pairs = 10,
        self.n_folds = 10,
        self.fold = []
        self.seed = 1234,
        self.max_length = 128,
        self.n_epochs = 3
        self.train_batch_size = 32
        self.valid_batch_size = 64,
        self.n_accumulate = 1,
        self.log_interval = 100,
        self.save_interval = 1000,
        self.eval_interval = 1000,
        self.eval_iters = 10,
        self.margin = 0.5,
        self.dropout = 0.2,
        self.hidden_size = 768,
        self.lr = 1e-4,
        self.T_max = 3000,
        self.min_lr = 1e-6,
        self.weight_decay = 1e-6
