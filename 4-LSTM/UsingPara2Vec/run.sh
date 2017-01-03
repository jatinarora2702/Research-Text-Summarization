source activate tensorflow
python2.7 seq2seq_attention.py \
   --mode=train \
   --article_key=article \
   --abstract_key=abstract  \
   --data_path=data/data/train/data-*  \
   --vocab_path=data/data/vocabulary  \
   --log_root=logs  \
   --train_dir=logs/train  \
   --truncate_input=False \
   --num_gpus=0  \
   --max_article_sentences=200  \
   --encoder_length=30 \
   --decoder_length=250 \
   --sent_embed_dimensions=400
