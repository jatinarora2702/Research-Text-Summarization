import gensim
import sys
import os
import codecs
import json
from preprocess import preprocess
import nltk.data
import multiprocessing

from collections import OrderedDict

model = gensim.models.doc2vec.Doc2Vec.load('Models/paragraph_DM.doc2vec')

model.save('Models_binary/paragraph_DM.doc2vec',pickle_protocol = 2)