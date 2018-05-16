#!/bin/bash


echo "done."
paste \
   /data1/ywang_wmt/$1-en/src.$1.tok.bpe \
   /data1/ywang_wmt/$1-en/trg.$1.tok.bpe \
   | sed "s/$(printf '\t')/ ||| /g" > /data1/ywang_wmt/$1-en/align.$1.tok.bpe 


echo "learn alignment"
cd  /data1/src/fast_align/build
./fast_align \
      -i  /data1/ywang_wmt/$1-en/align.$1.tok.bpe  \
      -v -r -p /data1/ywang_wmt/$1-en/prob.$1.tok.bpe \
      > /data1/ywang_wmt/$1-en/$1.alignment  

echo "Find the most probable translation for each word and write them to a file"

sort -k1,1 -k3,3gr /data1/ywang_wmt/$1-en/prob.$1.tok.bpe \
    | sort -k1,1 -u \
    > /data1/ywang_wmt/$1-en/dict.$1.tok.bpe 

python /data1/src/tools/clean_dict.py /data1/ywang_wmt/$1-en/dict.$1.tok.bpe \
                                     /data1/ywang_wmt/$1-en/trg.$1.tok.bpe \
                            > /data1/ywang_wmt/$1-en/dict.$1.tok.bpe.clean 

echo "all done"


python /data1/src/tools/compute_svd_trans.py \
            /data1/ywang_wmt/$1-en/fastText/$1.skipgram.vec \
            /data1/ywang_wmt/align/en.tok.bpe.skipgram.vec  \
            /data1/ywang_wmt/$1-en/dict.$1.tok.bpe.clean \
            /data1/ywang_wmt/$1-en/$1.bpe.map

echo "SVD done!"
