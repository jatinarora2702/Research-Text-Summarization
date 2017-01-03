# -*- coding: utf8 -*-

from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

import math

try:
    import numpy
except ImportError:
    numpy = None


from ._summarizer import AbstractSummarizer
from .._compat import Counter
from ..embeddings import embed_model


class LexRankEmbedSummarizer(AbstractSummarizer):
    """
    LexRank: Graph-based Centrality as Salience in Text Summarization
    Source: http://tangra.si.umich.edu/~radev/lexrank/lexrank.pdf
    """
    threshold = 0.1
    epsilon = 0.1
    _stop_words = frozenset()

    @property
    def stop_words(self):
        return self._stop_words

    @stop_words.setter
    def stop_words(self, words):
        self._stop_words = frozenset(map(self.normalize_word, words))

    def __call__(self, document, sentences_count):
        self._ensure_dependencies_installed()

        sentences_words = [self._to_words_set(s) for s in document.sentences]
        if not sentences_words:
            return tuple()

        #tf_metrics = self._compute_tf(sentences_words)
        #idf_metrics = self._compute_idf(sentences_words)

        model = embed_model.model()
        
        matrix = self._create_matrix(sentences_words, self.threshold, model)
        scores = self.power_method(matrix, self.epsilon)
        ratings = dict(zip(document.sentences, scores))

        return self._get_best_sentences(document.sentences, sentences_count, ratings)

    @staticmethod
    def _ensure_dependencies_installed():
        if numpy is None:
            raise ValueError("LexRank summarizer requires NumPy. Please, install it by command 'pip install numpy'.")

    def _to_words_set(self, sentence):
        words = map(self.normalize_word, sentence.words)
        return [w.lower() for w in words if w not in self._stop_words]   

    def _create_matrix(self, sentences, threshold, model):
        """
        Creates matrix of shape |sentences|×|sentences|.
        """
        # create matrix |sentences|×|sentences| filled with zeroes
        sentences_count = len(sentences)
        matrix = numpy.zeros((sentences_count, sentences_count))
        degrees = numpy.zeros((sentences_count, ))

        for row, sentence1 in enumerate(sentences):
            for col, sentence2 in enumerate(sentences):
            	score = 0
            	if(model.wmdistance(sentence1, sentence2) > 0):
            		score = 1 / model.wmdistance(sentence1, sentence2)
                matrix[row, col] = score #self._compute_cosine(sentence1, sentence2, tf1, tf2, idf_metrics)

                if matrix[row, col] > threshold:
                    matrix[row, col] = 1.0
                    degrees[row] += 1
                else:
                    matrix[row, col] = 0

        for row in range(sentences_count):
            for col in range(sentences_count):
                if degrees[row] == 0:
                    degrees[row] = 1

                matrix[row][col] = matrix[row][col] / degrees[row]

        return matrix

    @staticmethod
    def power_method(matrix, epsilon):
        transposed_matrix = matrix.T
        sentences_count = len(matrix)
        p_vector = numpy.array([1.0 / sentences_count] * sentences_count)
        lambda_val = 1.0

        while lambda_val > epsilon:
            next_p = numpy.dot(transposed_matrix, p_vector)
            lambda_val = numpy.linalg.norm(numpy.subtract(next_p, p_vector))
            p_vector = next_p

        return p_vector