# -*- coding:utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Author: Tuozhen
# Date: 2021/12/9
# Description:
# ----------------------------------------------------------------------------------------------------------------------
import time
import os
import numpy as np
import torch
import torch.nn as nn
from dataset import get_batch


def evaluate_step(args, model, dataloader, eval_iters=None):
    if eval_iters is None:
        eval_iters = args.eval_iters
    model.eval()
    loss = 0
    acc = 0
    iteration = 0
    start = time.time()

    with torch.no_grad():
        for step, data in enumerate(dataloader):
            if iteration == eval_iters:
                break
            more_toxic_ids, more_toxic_mask, less_toxic_ids, less_toxic_mask, targets = get_batch(args, data)

            more_toxic_outputs = model(more_toxic_ids, more_toxic_mask)
            less_toxic_outputs = model(less_toxic_ids, less_toxic_mask)

            loss += nn.MarginRankingLoss(margin=args.margin)(more_toxic_outputs, less_toxic_outputs, targets)
            # print(more_toxic_outputs.shape)
            # print(less_toxic_outputs.shape)
            # print(more_toxic_outputs - less_toxic_outputs > 0)
            # print((more_toxic_outputs - less_toxic_outputs > 0).sum())
            # time.sleep(1000)
            acc += (more_toxic_outputs - less_toxic_outputs > 0).sum()
            iteration += 1
    loss = loss / iteration / args.valid_batch_size
    acc = acc / iteration / args.valid_batch_size

    time_elapsed = time.time() - start
    print(f"***Evaluation***  |  [iteration] {args.eval_iters}   |  [loss] {round(loss.item(), 8)}   |  [acc] {round(acc.item(), 5)}  |  [total elapsed] {int(time_elapsed // 3600)}h, {int((time_elapsed % 3600) // 60)}m, {int((time_elapsed % 3600) % 60)}s")


def finetune_step(args, fold, epoch, model, optimizer, scheduler, train_dataloader, cv_valid_dataloader, valid_dataloader):
    start = time.time()

    n_consumption = 0
    running_loss = []
    iteration = 0
    total_iteration = len(train_dataloader) // args.n_accumulate
    print(f"total_iteration: {total_iteration}")

    for step, data in enumerate(train_dataloader):
        more_toxic_ids, more_toxic_mask, less_toxic_ids, less_toxic_mask, targets = get_batch(args, data)
        batch_size = more_toxic_ids.size(0)

        # train
        model.train()
        more_toxic_outputs = model(more_toxic_ids, more_toxic_mask)
        less_toxic_outputs = model(less_toxic_ids, less_toxic_mask)

        loss = nn.MarginRankingLoss(margin=args.margin)(more_toxic_outputs, less_toxic_outputs, targets)
        loss.backward()

        running_loss.append(loss.item() / batch_size)
        n_consumption += batch_size

        if (step + 1) % args.n_accumulate == 0:
            optimizer.step()
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

            iteration += 1

            if iteration % args.log_interval == 0 or iteration == total_iteration:
                time_elapsed = time.time() - start
                print(f"[iter] {iteration}/{total_iteration} | [consumption] {n_consumption} | [lr] {round(optimizer.param_groups[0]['lr'], 8)} | [loss] {round(np.mean(running_loss[-args.log_interval:]), 8)} | [elapsed] {int(time_elapsed // 3600)}h, {int((time_elapsed % 3600) // 60)}m, {int((time_elapsed % 3600) % 60)}s")

            if iteration % args.save_interval == 0 or iteration == total_iteration:
                PATH = os.path.join(args.checkpoint_path, "fold-" + str(fold))
                if not os.path.exists(PATH):
                    os.mkdir(PATH)
                PATH = PATH + f"-epoch{epoch}-{iteration}.bin"
                torch.save(model.state_dict(), PATH)

            if iteration % args.eval_interval == 0 or iteration == total_iteration:
                print("*** CV evaluation***")
                evaluate_step(args, model, cv_valid_dataloader)
                print("*** 2021 evaluation***")
                evaluate_step(args, model, valid_dataloader, eval_iters=10000)


def finetune(args, model, optimizer, scheduler, fold, train_dataloader, cv_valid_dataloader, valid_dataloader):
    if torch.cuda.is_available():
        print("[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))

    start = time.time()
    # best_model_wts = copy.deepcopy(model.state_dict())
    # best_epoch_loss = np.inf
    # history = defaultdict(list)

    for epoch in range(args.n_epochs):
        print(f"[epoch {epoch}]" + "-"*50)
        finetune_step(args, fold, epoch, model, optimizer, scheduler, train_dataloader, cv_valid_dataloader, valid_dataloader)

        # print("*** 2021 evaluation***")
        # evaluate_step(args, model, valid_dataloader, eval_iters=10000)

        # val_epoch_loss = valid_one_epoch(model, valid_loader, device=CONFIG['device'],
        #                                  epoch=epoch)
        #
        # history['Train Loss'].append(train_epoch_loss)
        # history['Valid Loss'].append(val_epoch_loss)
        #
        # # deep copy the model
        # if val_epoch_loss <= best_epoch_loss:
        #     print(f"Validation Loss Improved ({best_epoch_loss} ---> {val_epoch_loss})")
        #     best_epoch_loss = val_epoch_loss
        #     best_model_wts = copy.deepcopy(model.state_dict())
        #     PATH = f"Loss-Fold-{fold}.bin"
        #     torch.save(model.state_dict(), PATH)
        #     # Save a model file from the current directory
        #     print(f"Model Saved")

    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    # print("Best Loss: {:.4f}".format(best_epoch_loss))

    # load best model weights
    # model.load_state_dict(best_model_wts)
