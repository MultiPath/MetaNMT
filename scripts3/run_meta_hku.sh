#!/usr/bin/env bash
set -x
python meta_nmt5.py \
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
                --valid_batch_size 2000 \
                --inter_size 4\
                --inner_steps 1 \
                --valid_steps 4 \
                --valid_epochs 5 \
                --use_wo \
                -s ro -t en -a es fr it pt\
                --universal \
                --sequential_learning \
                --finetune_params 'emb_enc' \
                #--debug
                #--cross_meta_learning \
                #--cross_rate 0.5 \

                #--no_meta_training
                #--debug
                #--no_meta_training \
                #--debug
                #--debug \
                #> meta2.log 2>&1 & tail -f meta2.log
		#--debug
                # --debug

                #--cross_meta_learning \
                #--universal_options "refined_V" "armax" \
                
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

