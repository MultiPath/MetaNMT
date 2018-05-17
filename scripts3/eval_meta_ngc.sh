#!/usr/bin/env bash


python meta_eval5.py \
                --prefix [time] \
                --gpu $1 \
                --eval-every 1000 \
                --data_prefix "/data/" \
                --vocab_prefix "/data/meta_europarl/tensors2/" \
                --workspace_prefix "/result/metanmt_output/" \
                --load_vocab \
                --dataset meta_europarl \
                --tensorboard \
                --batch_size 2000 \
                --valid_batch_size 8000 \
                --inter_size 2 \
                --inner_steps 1 \
                --valid_steps 4 \
                --valid_epochs 10 \
                --use_wo \
                -s ${TARGET} -t en \
                --universal \
                --sequential_learning \
                --load_from ${MODEL} \
                --resume \
                --finetune_params $2 \
                --debug

                # --load_from "05.13_19.52.meta_europarl_default_ro-en-esptitfr_universal____4000_1" \
                #  --load_from "05.13_19.52.meta_europarl_default_ro-en-esptitfr_universal____4000_1" \



