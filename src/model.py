# -*- coding:utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Author: Tuozhen
# Date: 2021/12/8
# Description:
# ----------------------------------------------------------------------------------------------------------------------
import time

import torch.nn as nn
from transformers import AutoModel


class JigsawModel(nn.Module):
    def __init__(self, args, model_path):
        # set the class attributes
        super(JigsawModel, self).__init__()
        self.model = AutoModel.from_pretrained(model_path)
        self.drop = nn.Dropout(p=args.dropout)
        self.fc = nn.Linear(768, 1)

    def forward(self, ids, mask):
        out = self.model(input_ids=ids, attention_mask=mask,
                         output_hidden_states=False)
        out = self.drop(out[1])
        outputs = self.fc(out)
        return outputs
