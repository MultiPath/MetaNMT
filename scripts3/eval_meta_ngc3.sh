#!/usr/bin/env bash

python meta_eval5.py \
                --prefix [time] \
                --gpu $1 \
                --eval-every 1000 \
                --data_prefix "/result/data/" \
                --vocab_prefix "/result/data/meta_europarl/tensors2/" \
                --workspace_prefix "/result/metanmt_output/" \
                --load_vocab \
                --dataset meta_europarl \
                --tensorboard \
                --batch_size 2000 \
                --valid_batch_size 8000 \
                --inter_size 2 \
                --inner_steps 1 \
                --support_size $5 \
                --valid_epochs 16 \
                --use_wo \
                -s $3 -t en \
                --universal \
                --sequential_learning \
                --load_from $4 \
                --resume \
                --finetune_params $2 \
                --debug

                # --load_from "05.13_19.52.meta_europarl_default_ro-en-esptitfr_universal____4000_1" \
                #  --load_from "05.13_19.52.meta_europarl_default_ro-en-esptitfr_universal____4000_1" \



