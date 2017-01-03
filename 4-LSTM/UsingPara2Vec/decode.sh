source activate tensorflow
python2.7 seq2seq_attention.py \
   --mode=decode \
   --article_key=article \
   --abstract_key=abstract  \
   --data_path=data/data/test/data-*  \
   --vocab_path=data/data/vocabulary  \
   --log_root=logs  \
   --decode_dir=logs/decode  \
   --truncate_input=False \
   --num_gpus=0  \
   --max_article_sentences=200  \
   --encoder_length=30 \
   --decoder_length=250 \
   --sent_embed_dimensions=400 \
   --use_only_cpu=True
