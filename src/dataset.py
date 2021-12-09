# -*- coding:utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Author: Tuozhen
# Date: 2021/12/8
# Description:
# ----------------------------------------------------------------------------------------------------------------------
from torch.utils.data import Dataset, DataLoader
import torch


class JigsawDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.max_len = max_length
        self.tokenizer = tokenizer
        self.more_toxic = df['more_toxic'].values
        self.less_toxic = df['less_toxic'].values

    def __len__(self):
        return len(self.more_toxic)

    def __getitem__(self, index):
        more_toxic = self.more_toxic[index]
        less_toxic = self.less_toxic[index]
        inputs_more_toxic = self.tokenizer.encode_plus(
            more_toxic,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length'
        )
        inputs_less_toxic = self.tokenizer.encode_plus(
            less_toxic,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length'
        )
        target = 1

        more_toxic_ids = inputs_more_toxic['input_ids']
        more_toxic_mask = inputs_more_toxic['attention_mask']

        less_toxic_ids = inputs_less_toxic['input_ids']
        less_toxic_mask = inputs_less_toxic['attention_mask']

        return {
            'more_toxic_ids': torch.tensor(more_toxic_ids, dtype=torch.long),
            'more_toxic_mask': torch.tensor(more_toxic_mask, dtype=torch.long),
            'less_toxic_ids': torch.tensor(less_toxic_ids, dtype=torch.long),
            'less_toxic_mask': torch.tensor(less_toxic_mask, dtype=torch.long),
            'target': torch.tensor(target, dtype=torch.long)
        }


def train_dataloader_provider(args, train_df, tokenizer):
    train_dataset = JigsawDataset(
        train_df,
        tokenizer=tokenizer,
        max_length=args.max_length
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        num_workers=2,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )
    return train_dataloader


def valid_dataloader_provider(args, valid_df, tokenizer):
    valid_dataset = JigsawDataset(
        valid_df,
        tokenizer=tokenizer,
        max_length=args.max_length
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.valid_batch_size,
        num_workers=2,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )
    return valid_dataloader


def get_batch(args, data):
    more_toxic_ids = data['more_toxic_ids'].to(args.device, dtype=torch.long)
    more_toxic_mask = data['more_toxic_mask'].to(args.device, dtype=torch.long)
    less_toxic_ids = data['less_toxic_ids'].to(args.device, dtype=torch.long)
    less_toxic_mask = data['less_toxic_mask'].to(args.device, dtype=torch.long)
    targets = data['target'].to(args.device, dtype=torch.long)
    return more_toxic_ids, more_toxic_mask, less_toxic_ids, less_toxic_mask, targets
