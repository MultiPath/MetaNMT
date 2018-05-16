#!/usr/bin/env bash


python meta_eval2.py \
                --prefix [time] \
                --gpu $1 \
                --eval-every 1000 \
                --data_prefix "/data1/ywang/" \
                --workspace_prefix "/data0/workspace/metanmt_new/" \
                --vocab_prefix "/data1/ywang/meta_europarl/tensors2/" \
                --finetune_dataset "train.16000.0" \
                --load_vocab \
                --dataset meta_europarl \
                --tensorboard \
                --batch_size 1000 \
                --inter_size 4 \
                --inner_steps 1 \
                --valid_steps 4 \
                --valid_epochs 10 \
                --use_wo \
                -s zh -t en \
                --universal \
                --sequential_learning \
                --load_from "05.12_15.31.meta_europarl_subword_512_512_6_8_0.100_16000_universal__meta" \
                --resume \
                --debug
                # --load_from "05.13_19.52.meta_europarl_default_ro-en-esptitfr_universal____4000_1" \
                #  --load_from "05.13_19.52.meta_europarl_default_ro-en-esptitfr_universal____4000_1" \
