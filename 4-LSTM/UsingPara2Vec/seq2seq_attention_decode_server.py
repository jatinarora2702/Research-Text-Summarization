# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Module for running decoding server."""

import os
import time

import tensorflow as tf
import beam_search
import data
import json
import numpy as np

FLAGS = tf.app.flags.FLAGS


class BSDecoderForServer(object):
  """Beam search decoder."""

  def __init__(self, model, hps, vocab):
    """Beam search decoding.

    Args:
      model: The seq2seq attentional model.
      batch_reader: The batch data reader.
      hps: Hyperparamters.
      vocab: Vocabulary
    """
    self._model = model
    self._model.build_graph()
    self._hps = hps
    self._vocab = vocab
    self._saver = tf.train.Saver()
    self.load_model()

  def load_model(self):
    saver = self._saver
    self._sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    sess = self._sess
    ckpt_state = tf.train.get_checkpoint_state(FLAGS.log_root)
    if not (ckpt_state and ckpt_state.model_checkpoint_path):
      tf.logging.info('No model to decode yet at %s', FLAGS.log_root)
      return False

    tf.logging.info('checkpoint path %s', ckpt_state.model_checkpoint_path)
    ckpt_path = os.path.join(
        FLAGS.log_root, os.path.basename(ckpt_state.model_checkpoint_path))
    tf.logging.info('renamed checkpoint path %s', ckpt_path)
    saver.restore(sess, ckpt_path)


  def update_session(self):
    self._sess.close()
    self.load_model()

  def DecodeOne(self, input_article):
    sess = self._sess
    (article, article_len, origin_article) = self._convertInputToModelTensor(input_article)
      
    bs = beam_search.BeamSearch(
        self._model, FLAGS.beam_size,
        self._vocab.WordToId(data.SENTENCE_START),
        self._vocab.WordToId(data.SENTENCE_END),
        self._hps.dec_timesteps)

    article_cp = [article]*self._hps.batch_size
    article_len_cp = [article_len]*self._hps.batch_size


    best_beam = bs.BeamSearch(sess, article_cp, article_len_cp)[0]
    decode_output = [int(t) for t in best_beam.tokens[1:]]
    summary = self._DecodeBatch(decode_output)
    print decode_output, summary
    return summary

  def _DecodeBatch(self, output_ids):
    """Convert id to words and writing results.

    Args:
      article: The original article string.
      abstract: The human (correct) abstract string.
      output_ids: The abstract word ids output by machine.
    """
    decoded_output = ' '.join(data.Ids2Words(output_ids, self._vocab))
    end_p = decoded_output.find(data.SENTENCE_END, 0)
    if end_p != -1:
      decoded_output = decoded_output[:end_p]
    return decoded_output.strip()


  def _convertInputToModelTensor(self, article):
    print(np.shape(article))

    enc_inputs = np.reshape(article, [-1, self._hps.sent_embed_dimensions]).tolist()

    print(np.shape(enc_inputs))

    pad_para = [0] * self._hps.sent_embed_dimensions
    

    if (len(enc_inputs) > self._hps.enc_timesteps):
      tf.logging.warning('Truncating the example - too long.\nenc:%d\n',
                         len(enc_inputs))

    if len(enc_inputs) > self._hps.enc_timesteps:
          enc_inputs = enc_inputs[:self._hps.enc_timesteps]

    # Now len(enc_inputs) should be <= enc_timesteps, and

    enc_input_len = len(enc_inputs)

    # Pad if necessary
    while len(enc_inputs) < self._hps.enc_timesteps:
      enc_inputs.append(pad_para)

    return (enc_inputs, enc_input_len, "")



  
