# -*- coding:utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Author: Tuozhen
# Date: 2021/12/8
# Description:
# ----------------------------------------------------------------------------------------------------------------------
import torch
import numpy as np
import argparse


def set_seed(seed=1234):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default='../input/jigsaw-toxic-severity-rating/validation_data.csv',
                        type=str, required=False, help="")
    parser.add_argument("--modified_train_path", default='../input/jigsawmodify-train-data/modified_train.csv',
                        type=str, required=False, help="")
    parser.add_argument("--valid_path", default='../input/jigsaw-toxic-severity-rating/validation_data.csv',
                        type=str, required=False, help="")
    parser.add_argument("--eval_model", nargs='+', default='',
                        type=str, required=False, help="")
    parser.add_argument("--model_path", default='../input/jigsaw-toxic-severity-rating/validation_data.csv',
                        type=str, required=False, help="")
    parser.add_argument("--checkpoint_path", default='../input/jigsaw-toxic-severity-rating/validation_data.csv',
                        type=str, required=False, help="")
    parser.add_argument("--pos_neg_pairs", default=10,
                        type=int, required=False, help="1:k")
    parser.add_argument("--n_folds", default=10,
                        type=int, required=False, help="")
    parser.add_argument('--fold', nargs='+', default=list(range(10)),
                        type=int, required=False, help="")
    parser.add_argument("--seed", default=1234,
                        type=int, required=False, help="")
    parser.add_argument("--max_length", default=128,
                        type=int, required=False, help="")
    parser.add_argument("--n_epochs", default=3,
                        type=int, required=False, help="")
    parser.add_argument("--train_batch_size", default=32,
                        type=int, required=False, help="")
    parser.add_argument("--valid_batch_size", default=64,
                        type=int, required=False, help="")
    parser.add_argument("--n_accumulate", default=1,
                        type=int, required=False, help="")
    parser.add_argument("--log_interval", default=100,
                        type=int, required=False, help="")
    parser.add_argument("--save_interval", default=1000,
                        type=int, required=False, help="")
    parser.add_argument("--eval_interval", default=1000,
                        type=int, required=False, help="")
    parser.add_argument("--eval_iters", default=10,
                        type=int, required=False, help="")
    parser.add_argument("--margin", default=0.5,
                        type=float, required=False, help="MarginRankingLoss")
    parser.add_argument("--dropout", default=0.2,
                        type=float, required=False, help="")
    parser.add_argument("--hidden_size", default=768,
                        type=int, required=False, help="")
    parser.add_argument("--lr", default=1e-4,
                        type=float, required=False, help="")
    parser.add_argument("--T_max", default=3000,
                        type=int, required=False, help="lr_scheduler Maximum number of iterations.")
    parser.add_argument("--min_lr", default=1e-6,
                        type=float, required=False, help="lr_scheduler Minimum learning rate.")
    parser.add_argument("--weight_decay", default=1e-6,
                        type=float, required=False, help="")
    args = parser.parse_args()

    return args
