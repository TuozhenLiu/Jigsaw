# -*- coding:utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Author: Tuozhen
# Date: 2021/12/8
# Description:
# ----------------------------------------------------------------------------------------------------------------------
import os
import pandas as pd
import numpy as np
import torch
from torch.optim import lr_scheduler
from transformers import AutoTokenizer
from transformers.optimization import Adafactor
from utils import set_seed, get_args
from model import JigsawModel
from dataset import train_dataloader_provider, valid_dataloader_provider
from train import finetune


def modify_train_df(args):
    if os.path.exists(args.modified_train_path):
        modified_df = pd.read_csv(args.modified_train_path)
        return modified_df

    df_test = pd.read_csv(os.path.join(args.train_path, "test.csv"))
    df_test_labels = pd.read_csv(os.path.join(args.train_path, "test_labels.csv"))
    df_test_labels = df_test_labels.replace(-1, 0)
    df_test = pd.concat([df_test["comment_text"], df_test_labels], axis=1)
    df_train = pd.read_csv(os.path.join(args.train_path, "train.csv"))
    df = pd.concat([df_train, df_test], axis=0)

    df["toxic_score"] = df.apply(lambda x: x['toxic'] * 2 + x["severe_toxic"] * 3 + x['obscene'] + x['threat'] + x["insult"] + x["identity_hate"], axis=1)

    more_toxic = []
    less_toxic = []
    zero_df = df[df.toxic_score == 0].reset_index(drop=True)

    print(f"original train samples: {df.shape[0] - zero_df.shape[0]}")

    for score in range(3, df["toxic_score"].max() + 1):
        pos_df = df[df.toxic_score == score]
        neg_df = df[(df.toxic_score > 0) & (df.toxic_score < score - 1)].reset_index(drop=True)
        if pos_df.shape[0] == 0:
            continue
        for pos in pos_df["comment_text"]:
            more_toxic.extend([pos] * args.pos_neg_pairs)
            neg_list = neg_df.loc[np.random.choice(neg_df.shape[0], args.pos_neg_pairs // 2, replace=False), "comment_text"].tolist()
            less_toxic.extend(neg_list)
            zero_list = zero_df.loc[np.random.choice(zero_df.shape[0], args.pos_neg_pairs // 2, replace=False), "comment_text"].tolist()
            less_toxic.extend(zero_list)
    for score in range(1, 3):
        pos_df = df[df["toxic_score"] == score]
        if pos_df.shape[0] == 0:
            continue
        for pos in pos_df["comment_text"]:
            more_toxic.extend([pos] * args.pos_neg_pairs)
            zero_list = zero_df.loc[np.random.choice(zero_df.shape[0], args.pos_neg_pairs, replace=False), "comment_text"].tolist()
            less_toxic.extend(zero_list)

    modified_df = pd.DataFrame({"more_toxic": more_toxic, "less_toxic": less_toxic})
    # modified_df.to_csv(os.path.join(args.train_path, "modified_train.csv"), index=False)
    print(f"modified train samples: {modified_df.shape[0]}")
    return modified_df


if __name__ == '__main__':
    print(f"====== Initializing ======")
    args = get_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if not os.path.exists(args.checkpoint_path):
        os.mkdir(args.checkpoint_path)

    train_df = modify_train_df(args)
    valid_df = pd.read_csv(args.valid_path)

    k_fold_indexes = np.array_split(np.random.permutation(np.arange(train_df.shape[0])), args.n_folds)

    for fold in args.fold:
        print(f"====== Fold: {fold} ======")
        train = train_df[~train_df.index.isin(k_fold_indexes[fold])].reset_index(drop=True)
        cv_valid = train_df[train_df.index.isin(k_fold_indexes[fold])].reset_index(drop=True)
        train_loader = train_dataloader_provider(args, train, tokenizer)
        cv_valid_loader = valid_dataloader_provider(args, cv_valid, tokenizer)
        valid_loader = valid_dataloader_provider(args, valid_df, tokenizer)

        model = JigsawModel(args)
        model.to(args.device)

        optimizer = Adafactor(
            model.parameters(),
            lr=args.lr,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=None,
            weight_decay=args.weight_decay,
            relative_step=False,
            scale_parameter=True,
            warmup_init=False
        )
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.T_max,
            eta_min=args.min_lr
        )

        finetune(args, model, optimizer, scheduler, fold, train_loader, cv_valid_loader, valid_loader)
