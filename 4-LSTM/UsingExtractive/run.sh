source activate tensorflow
python2.7 seq2seq_attention.py \
   --mode=train \
   --article_key=article \
   --abstract_key=abstract  \
   --data_path=data/data_micro/train/data-*  \
   --vocab_path=data/data_micro/vocabulary  \
   --log_root=logs_micro \
   --train_dir=logs_micro/train  \
   --truncate_input=False \
   --num_gpus=0  \
   --max_article_sentences=2000  \
   --encoder_length=300 \
   --decoder_length=50
