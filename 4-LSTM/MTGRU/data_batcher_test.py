import sys
import time

import tensorflow as tf
import batch_reader
import data
import seq2seq_attention_decode

import seq2seq_attention_model


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_path',
                           'data/sample_data/data', 'Path expression to tf.Example.')
tf.app.flags.DEFINE_string('vocab_path',
                           'data/sample_data/vocab', 'Path expression to text vocabulary file.')
tf.app.flags.DEFINE_string('article_key', 'article',
                           'tf.Example feature key for article.')
tf.app.flags.DEFINE_string('abstract_key', 'abstract',
                           'tf.Example feature key for abstract.')
tf.app.flags.DEFINE_string('log_root', '', 'Directory for model root.')
tf.app.flags.DEFINE_string('train_dir', '', 'Directory for train.')
tf.app.flags.DEFINE_string('eval_dir', '', 'Directory for eval.')
tf.app.flags.DEFINE_string('decode_dir', '', 'Directory for decode summaries.')
tf.app.flags.DEFINE_string('mode', 'train', 'train/eval/decode mode')
tf.app.flags.DEFINE_integer('max_run_steps', 10000000,
                            'Maximum number of run steps.')
tf.app.flags.DEFINE_integer('max_article_sentences', 100,
                            'Max number of first sentences to use from the '
                            'article')
tf.app.flags.DEFINE_integer('max_abstract_sentences', 100,
                            'Max number of first sentences to use from the '
                            'abstract')
tf.app.flags.DEFINE_integer('beam_size', 4,
                            'beam size for beam search decoding.')
tf.app.flags.DEFINE_integer('eval_interval_secs', 60, 'How often to run eval.')
tf.app.flags.DEFINE_integer('checkpoint_secs', 60, 'How often to checkpoint.')
tf.app.flags.DEFINE_bool('use_bucketing', False,
                         'Whether bucket articles of similar length.')
tf.app.flags.DEFINE_bool('truncate_input', True,
                         'Truncate inputs that are too long. If False, '
                         'examples that are too long are discarded.')
tf.app.flags.DEFINE_integer('num_gpus', 0, 'Number of gpus used.')
tf.app.flags.DEFINE_integer('random_seed', 111, 'A seed value for randomness.')
tf.app.flags.DEFINE_integer('encoder_length', 120, 'Encoder length')
tf.app.flags.DEFINE_integer('decoder_length', 30, 'Decoder length')




vocab = data.Vocab(FLAGS.vocab_path, 1000000)
# Check for presence of required special tokens.
assert vocab.WordToId(data.PAD_TOKEN) > 0
assert vocab.WordToId(data.UNKNOWN_TOKEN) >= 0
assert vocab.WordToId(data.SENTENCE_START) > 0
assert vocab.WordToId(data.SENTENCE_END) > 0

batch_size = 4
hps = seq2seq_attention_model.HParams(
      mode=FLAGS.mode,  # train, eval, decode
      min_lr=0.01,  # min learning rate.
      lr=0.15,  # learning rate
      batch_size=batch_size,
      enc_layers=4,
      enc_timesteps=FLAGS.encoder_length,
      dec_timesteps=FLAGS.decoder_length,
      min_input_len=5,  # discard articles/summaries < than this
      num_hidden=256,  # for rnn cell
      emb_dim=128,  # If 0, don't use embedding
      max_grad_norm=2,
      num_softmax_samples=4096)  # If 0, no sampled softmax.


batcher = batch_reader.Batcher(
      FLAGS.data_path, vocab, hps, FLAGS.article_key,
      FLAGS.abstract_key, FLAGS.max_article_sentences,
      FLAGS.max_abstract_sentences, bucketing=FLAGS.use_bucketing,
      truncate_input=FLAGS.truncate_input)


import pprint
pp = pprint.PrettyPrinter(indent=4)
while(True):
	batch = batcher.NextBatch()
	pp.pprint(batch[-1])
	r = input()
exit(0)

