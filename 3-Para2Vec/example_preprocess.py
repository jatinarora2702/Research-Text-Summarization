import nltk
import tensorflow as tf
import os
from collections import Counter
import json
import re
import codecs

import test_embeddings
# Special tokens
PARAGRAPH_START = '<p>'
PARAGRAPH_END = '</p>'
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'
DOCUMENT_START = '<d>'
DOCUMENT_END = '</d>'

UNKNOWN_TOKEN = '<UNK>'

MIN_FREQ = 4


indi_len = [0, 0]
sum_len = [0,0]
nodocs = [0]

def sentence_tokenizer():
	# Train a Punkt tokenizer if required
	return nltk.data.load('tokenizers/punkt/english.pickle')


def word_tokenizer():
	return nltk.tokenize.treebank.TreebankWordTokenizer()

def remove_digits(parse):
	return re.sub(r'\d', '#', parse)

sentence_segmenter = sentence_tokenizer()
word_tok = word_tokenizer()

words = Counter()

def process_sentence(sentence):
	## Add more processing if required
    tokens = [SENTENCE_START]
    tokens.extend(word_tok.tokenize(sentence))
    tokens.append(SENTENCE_END)
    return tokens

def para_to_tokens(paragraph):
	return [token for sent in sentence_segmenter.tokenize(paragraph.strip())
        for token in process_sentence(sent)]

def article_to_tokens(article):
    return [token for sent in article.split('\n')
        for token in process_sentence(sent)]

def getDocumentFromJson(document_json):
    docuemnt = [DOCUMENT_START]
    for key in document_json:
        section = document_json[key]
        section_name = key
        tokens = para_to_tokens(section)
        docuemnt.extend(tokens)
    docuemnt.append(DOCUMENT_END)
    return " ".join(docuemnt)

def get_doc_from_tokens(tokens):
    docuemnt = [DOCUMENT_START] + tokens + [DOCUMENT_END]
    return " ".join(docuemnt)



def pre_example_process(text):
    #print(text)
    text = text.replace("< ref >", "||REF||")
    return remove_digits(text).lower()


def parse_line_to_example(name, abstract):

    
    with codecs.open(abstract_file, "r", "utf-8") as abst:
        abstract = pre_example_process(getDocumentFromJson({"abstract" :abstract}))
    toks = abstract.split(" ")
    indi_len[1] = max( indi_len[1], len(toks))
    sum_len[1] = sum_len[1] + len(toks)
    words.update(toks)
    nodocs[0] = nodocs[0] + 1
    article_rep = test_embeddings.get_embedding(name)
    article_rep = tf.reshape(article_rep, [-1])
    return tf.train.Example(
            features = tf.train.Features(
                feature = {
                    'article': tf.train.Feature(float_list=tf.train.FloatList(value=article_rep)),
                    'abstract': tf.train.Feature(bytes_list=tf.train.BytesList(value=[abstract.encode('utf-8')])),
                    'name': tf.train.Feature(bytes_list=tf.train.BytesList(value=[name.encode("utf-8")]))
                }
            )
        )



def create_data_files(to_path):
    papers = test_embeddings.init()
    fileno = 1
    writer = tf.python_io.TFRecordWriter(os.path.join(to_path,"data-"+str(fileno)))
    cnt = 0
    MAX_LEN_FILE = 100
    for paper in papers:
        abstract = paper['sum']
        name = paper['id']
        writer.write(parse_line_to_example(name, abstract).SerializeToString())
        cnt = cnt + 1
        if(cnt % MAX_LEN_FILE == 0):
            cnt = 0
            fileno = fileno + 1
            writer.close()
            writer = tf.python_io.TFRecordWriter(os.path.join(to_path,"data-"+str(fileno)))

    vocab = open(os.path.join(to_path, "vocabulary"), 'w')
    n = 0
    vocab_list = words.most_common()
    for word, count in vocab_list:
        if(count <  MIN_FREQ): break
        n = n +1
        #if(word[0] == '#'): continue
        #print >>vocab, word.encode('utf-8'), count
        print(word, count, file=vocab)
        
    unk = vocab_list[n:-1]
    sum = 0
    for _, cnt in unk:
        sum = sum + cnt
    #print >>vocab, UNKNOWN_TOKEN, sum
    print(UNKNOWN_TOKEN, sum, file=vocab)
    vocab.close()




create_data_files("output")
print(indi_len, sum_len, nodocs)

	
