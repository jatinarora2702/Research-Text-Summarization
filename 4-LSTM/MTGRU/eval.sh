source activate tensorflow
python2.7 seq2seq_attention.py   \
	--mode=eval   \
	--article_key=article   \
	--abstract_key=abstract  \
	--data_path=data/data_micro/validation/data-*  \
	--vocab_path=data/data_micro/vocabulary  \
	--log_root=logs_micro  \
	--eval_dir=logs_micro/eval \
	--truncate_input=False \
	--num_gpus=0  \
	--max_article_sentences=2000  \
	--encoder_length=300 \
	--decoder_length=50 \
	--use_only_cpu=true
