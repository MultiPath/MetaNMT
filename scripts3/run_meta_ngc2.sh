#!/usr/bin/env bash
LOGID=$2
python meta_nmt3.py \
                --prefix [time] \
                --gpu $1 \
                --eval-every 1000 \
                --data_prefix "/data/" \
                --vocab_prefix "/data/meta/tensors2/" \
                --workspace_prefix "/result/metanmt_output/" \
                --finetune_dataset "finetune.600.tok" \
                --load_vocab \
                --dataset meta \
                --tensorboard \
                --batch_size 1000 \
                --inter_size 4 \
                --inner_steps 1 \
                --valid_steps 4 \
                --valid_epochs 5 \
                --use_wo \
                -s ro -t en -a es \
                --universal \
                --sequential_learning \

                #--cross_meta_learning \
                #> /result/metanmt_output/${LOGID} 2>&1 & tail -f /result/metanmt_output/${LOGID}
		        #--debug
		        #--meta_approx_2nd \
                #--approx_lr 0.000001 \
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

