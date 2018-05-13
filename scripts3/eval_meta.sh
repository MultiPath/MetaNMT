#!/usr/bin/env bash


python meta_eval.py \
                --prefix [time] \
                --gpu $1 \
                --eval-every 1000 \
                --data_prefix "/data1/ywang/" \
                --workspace_prefix "/data0/workspace/metanmt_new/" \
                --vocab_prefix "/data1/ywang/meta_europarl/tensors2/" \
                --finetune_dataset "finetune.600.tok" \
                --load_vocab \
                --dataset meta_europarl \
                --tensorboard \
                --batch_size 1000 \
                --inter_size 4 \
                --inner_steps 1 \
                --valid_steps 4 \
                --valid_epochs 10 \
                --use_wo \
                -s ro -t en \
                --universal \
                --sequential_learning \
                --load_from "05.12_15.31.meta_europarl_subword_512_512_6_8_0.100_16000_universal__meta" \
                --resume
                --debug