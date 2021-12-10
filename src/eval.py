# -*- coding:utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Author: Tuozhen
# Date: 2021/12/10
# Description:
# ----------------------------------------------------------------------------------------------------------------------
import time
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer
from utils import set_seed, get_args
from model import JigsawModel
from dataset import eval_dataloader_provider, get_eval_batch


if __name__ == '__main__':
    print(f"====== Initializing ======")
    args = get_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    eval_model_list = args.eval_model
    eval_df = pd.read_csv(args.valid_path)

    print(f"====== Eval ======")
    final_preds = []
    start = time.time()
    for i, eval_model in enumerate(eval_model_list):
        print(f"Getting predictions for model {i}: [eval_model]")
        eval_loader = eval_dataloader_provider(args, eval_df, tokenizer)

        model = JigsawModel(args)
        model.to(args.device)
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(eval_model))
        else:
            model.load_state_dict(torch.load(eval_model, map_location=torch.device('cpu')))

        model.eval()

        single_preds = []
        with torch.no_grad():
            for data in eval_loader:

                text_token_ids, text_token_mask = get_eval_batch(args, data)

                outputs = model(text_token_ids, text_token_mask)
                single_preds.extend(outputs.view(-1).cpu().detach().numpy().tolist())
        final_preds.append(single_preds)
        time_elapsed = time.time() - start
        print(f"***Evaluation***  |  [total elapsed] {int(time_elapsed // 3600)}h, {int((time_elapsed % 3600) // 60)}m, {int((time_elapsed % 3600) % 60)}s")

    final_preds = np.array(final_preds)
    # add scaling?
    final_preds = np.mean(final_preds, axis=0)
    final_preds = (final_preds - final_preds.min()) / (final_preds.max() - final_preds.min())
    df_sub = pd.DataFrame()
    df_sub["comment_id"] = eval_df["comment_id"]
    df_sub["score"] = final_preds
    df_sub.to_csv("submission.csv", index=False)
