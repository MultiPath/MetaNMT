#!/usr/bin/env bash

python meta_select.py \
                --prefix [time] \
                --gpu $1 \
                --eval-every 1000 \
                --data_prefix "/data1/ywang/" \
                --vocab_prefix "/data1/ywang/meta_europarl/tensors2/" \
                --workspace_prefix "/data0/workspace/metanmt_new/" \
                --load_vocab \
                --dataset meta_europarl \
                --tensorboard \
                --batch_size 2000 \
                --valid_batch_size 4000 \
                --inter_size 4\
                --inner_steps 1 \
                --valid_steps 4 \
                --valid_epochs 5 \
                --use_wo \
                -s $3 -t en \
                --universal \
                --sequential_learning \
                --load_from 05.14_23.54.meta_europarl_default_ro-en-esfritptderu_universal____4000_1_ \
                --resume \
                --finetune_params $2 \
                --debug

                # --load_from "05.13_19.52.meta_europarl_default_ro-en-esptitfr_universal____4000_1" \
                #  --load_from "05.13_19.52.meta_europarl_default_ro-en-esptitfr_universal____4000_1" \


# lv selected model
#05.14_23.54.meta_europarl_default_ro-en-esfritptderu_universal____4000_1_.iter=80000
# [[0.0633, 10000], [0.075, 20000], [0.0791, 30000], [0.0769, 40000], [0.0831, 50000], [0.079, 60000], [0.0809, 70000], [0.0823, 80000], [0.0802, 90000], [0.0807, 100000], [0.0782, 110000], [0.0779, 120000]]