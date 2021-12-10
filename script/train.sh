python ../input/jigsawscript/main.py \
  --train_path ../input/jigsaw-toxic-comment-classification-challenge \
  --valid_path ../input/jigsaw-toxic-severity-rating/validation_data.csv \
  --modified_train_path ../input/jigsawmodify-train-data/modified_train.csv \
  --model_path ../input/roberta-transformers-pytorch/roberta-base \
  --checkpoint_path ./checkpoint \
  --seed 1234 \
  --fold 0 \
  --n_epochs 3 \
  --pos_neg_pairs 10 \
  --train_batch_size 64 \
  --valid_batch_size 128 \
  --n_accumulate 1 \
  --lr 3e-4 \
  --min_lr 1e-5 \
  --T_max 3000 \
  --log_interval 100 \
  --eval_interval 1000 \
  --eval_iters 100

python ../input/jigsawscript/main.py \
  --train_path ../input/jigsaw-toxic-comment-classification-challenge \
  --valid_path ../input/jigsaw-toxic-severity-rating/validation_data.csv \
  --modified_train_path xxx \
  --model_path ../input/roberta-transformers-pytorch/roberta-large \
  --checkpoint_path ./checkpoint \
  --seed 2823 \
  --fold 0 \
  --n_epochs 1 \
  --pos_neg_pairs 10 \
  --train_batch_size 16 \
  --valid_batch_size 32 \
  --n_accumulate 4 \
  --lr 3e-4 \
  --min_lr 1e-5 \
  --T_max 3000 \
  --log_interval 10 \
  --eval_interval 1000 \
  --eval_iters 100 \
  --hidden_size 1024

python ../input/jigsawscript/eval.py \
  --model_path ../input/roberta-transformers-pytorch/roberta-base \
  --valid_path ../input/jigsaw-toxic-severity-rating/comments_to_score.csv \
  --eval_model ../input/robertabase3000iters/checkpoint/fold-0-epoch0-1000.bin \
               ../input/robertabase3000iters/checkpoint/fold-0-epoch0-2000.bin \
               ../input/robertabase3000iters/checkpoint/fold-0-epoch0-3000.bin \
               ../input/robertabase3000iters/checkpoint/fold-1-epoch0-1000.bin \
               ../input/robertabase3000iters/checkpoint/fold-1-epoch0-2000.bin \
               ../input/robertabase3000iters/checkpoint/fold-1-epoch0-3000.bin \
  --seed 1234 \
  --valid_batch_size 128