from SimpleXMLRPCServer import SimpleXMLRPCServer
from SimpleXMLRPCServer import SimpleXMLRPCRequestHandler

from test_embeddings import get_embedding

import gensim

import seq2seq_attention_decode_server
import os
import nltk
import re
import codecs
import numpy as np
from xmlrpclib import ServerProxy
import pickle

# Special tokens
PARAGRAPH_START = '<p>'
PARAGRAPH_END = '</p>'
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'
DOCUMENT_START = '<d>'
DOCUMENT_END = '</d>'

UNKNOWN_TOKEN = '<UNK>'

# Restrict to a particular path.
class RequestHandler(SimpleXMLRPCRequestHandler):
    rpc_paths = ('/RPC2',)

def sentence_tokenizer():
	# Train a Punkt tokenizer if required
	return nltk.data.load('tokenizers/punkt/english.pickle')


def word_tokenizer():
	return nltk.tokenize.treebank.TreebankWordTokenizer()

def remove_digits(parse):
	return re.sub(r'\d', '#', parse)

sentence_segmenter = sentence_tokenizer()
word_tok = word_tokenizer()

def get_doc_from_tokens(tokens):
    docuemnt = [DOCUMENT_START] + tokens + [DOCUMENT_END]
    return " ".join(docuemnt)

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

def pre_example_process(text):
    #print(text)
    text = text.replace("< ref >", "||REF||")
    return remove_digits(text).lower()

def article_to_tokens(article):
    return [token for sent in article.split('\n')
        for token in process_sentence(sent)]

para_model = None


class DecodeServer(object):

	def __init__(self, decoder_for_server, port):
		# Create server
		global para_model
		self.server = SimpleXMLRPCServer(("0.0.0.0", port),
		                            requestHandler=RequestHandler)
		self.server.register_introspection_functions()

		self.decoder = decoder_for_server

		self.server.register_function(self.summarize)

		self.server.register_function(self.update_session)

		self.server.register_function(self.summarize_with_string)

		self.embed_proxy = ServerProxy('http://10.5.18.109:11000')

		print self.embed_proxy.system.listMethods()

		#para_model = gensim.models.doc2vec.Doc2Vec.load_word2vec_format('data/Models_binary/paragraph_DM.doc2vec', binary=True)

		print("Server running at port "+str(port))

		self.server.serve_forever()




	def update_session(self):
		self.decoder.update_session()
		return "done"


	def summarize_with_string(self, latex):
		inp_file = codecs.open("/home/bt1/13CS10060/snlp16/Test_Data/Input/temp.tex", "w", "utf-8")
		inp_file.write(latex)
		inp_file.close()

		return self.summarize("temp.tex")

	def summarize(self, filename):

		global para_model

		summary = ""
		## Preprocess article to generate input for the model

		# Parse Latex File - Convert to Text
		os.system('python2 ../../../Codes/pylatexenc/parse_latex_file.py '+filename)

		# Parse Text File - Convert to Dict (Section: Text)
		os.system('python2 ../../../Codes/pylatexenc/parse_text_file.py '+filename)

		# Generate ParaVec
		try:
			print self.embed_proxy.system.listMethods()
		except:
			print "ehere"
		print("here1")

		try:
			pkl_filename = self.embed_proxy.embed(filename)
		except:
			print "here am i"

		print("here21", pkl_filename)
		pkl_file = open(pkl_filename,'rb')

		(para_vecs, actual_abstract) = pickle.load(pkl_file)

		print("here2")
		

		# path - /home/bt1/13CS10060/snlp16/Test_Data/data/summary/lex-rank/100

		abstract = pre_example_process(getDocumentFromJson({"abstract" :actual_abstract}))

		print(np.shape(para_vecs))
		article_rep = np.reshape(para_vecs, [-1]).astype('float')
		print(np.shape(article_rep))

		print("here3")


		#print(article)

		summary = self.decoder.DecodeOne(article_rep)

		#summary = ""

		## Post process model summary if required
		print(actual_abstract, summary)
		return (actual_abstract,summary)


## Test Case

# source = codecs.open("my.tex", "r","utf-8")
# print( summarize_with_string(source.read()))
# source.close()
#summarize('comm_magazine_FDMIMO_overview.tex')

