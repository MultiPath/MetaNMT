#!/usr/bin/env bash
CUDA_LAUNCH_BLOCKING=1 python meta_nmt3.py \
                --prefix [time] \
                --gpu $1 \
                --eval-every 500 \
                --data_prefix "/data1/ywang/" \
                --vocab_prefix "/data1/data/uninmt/parl/tensors/" \
                --workspace_prefix "/data0/workspace/metanmt_new/" \
                --finetune_dataset "finetune.600.tok" \
                --load_vocab \
                --dataset meta_europarl \
                --tensorboard \
                --batch_size 500 \
                --inter_size 8 \
                --inner_steps 1 \
                --valid_steps 4 \
                --valid_epochs 5 \
                --use_wo \
                -s ro -t en -a es pt it fr \
                --universal \
                --sequential_learning \
                --debug
                # --meta_approx_2nd \
                # --approx_lr 0.000001 \
                # --debug


                #--universal_options "refined_V" "argmax" \
                
                #--debug \
                #--debug
                # --debug \

                # --debug \
                # --universal_options "argmax" \
                #--debug
                #--sequential_learning \

                #--debug
                #--no_meta_training \
                # --debug
                #--sequential_learning \
                #--debug
                #--resume \
                #--sequential_learning \
                #--load_from 04.12_22.12.meta_europarl_subword_512_512_6_8_0.100_16000_universal__meta
                #--load_from 04.12_22.11.meta_europarl_subword_512_512_6_8_0.100_16000_universal__meta \
                #--resume \
                #--sequential_learning \
                #--debug \
                #--debug \
                #--share_universal_embedding \
                # --load_dataset \
                # --debug \
                # --dataset europarl_6k_fv \

