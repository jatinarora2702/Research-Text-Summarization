source activate tensorflow
python2.7 seq2seq_attention.py   \
	--mode=decode  \
	--article_key=article   \
	--abstract_key=abstract  \
	--data_path=data/data_micro/test/data-*  \
	--vocab_path=data/data_micro/vocabulary  \
	--log_root=logs_micro  \
	--decode_dir=logs_micro/decode \
	--truncate_input=False \
	--num_gpus=0  \
	--max_article_sentences=2000  \
	--encoder_length=400 \
	--decoder_length=80 \
    	--use_only_cpu=True
