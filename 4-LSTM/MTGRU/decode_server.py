from SimpleXMLRPCServer import SimpleXMLRPCServer
from SimpleXMLRPCServer import SimpleXMLRPCRequestHandler

from parser import get_paras

import seq2seq_attention_decode_server
import os
import nltk
import re
import codecs

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

class DecodeServer(object):

	def __init__(self, decoder_for_server, port):
		# Create server
		self.server = SimpleXMLRPCServer(("0.0.0.0", port),
		                            requestHandler=RequestHandler)
		self.server.register_introspection_functions()

		self.decoder = decoder_for_server

		self.server.register_function(self.summarize)

		self.server.register_function(self.update_session)

		self.server.register_function(self.summarize_with_string)

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

		summary = ""
		## Preprocess article to generate input for the model

		# Parse Latex File - Convert to Text
		os.system('python2 ../../../Codes/pylatexenc/parse_latex_file.py '+filename)

		# Parse Text File - Convert to Dict (Section: Text)
		os.system('python2 ../../../Codes/pylatexenc/parse_text_file.py '+filename)

		# Parse Dict - Dict to File

		print("Here1")
		(para_list, actual_abstract) = get_paras(filename)

		print("JHere2")

		print(len(para_list), actual_abstract)
		print(para_list[0])


		article_list = []
		for para in para_list:
			article = pre_example_process(getDocumentFromJson({"abstract" :para}))
			if(len(article.split(" ")) < 4):
				continue
			article_list.append(article)

		print("Here3")

		if(len(article_list) < 1):
			return (actual_abstract, " ")

		#print(article)

		summary_list = []
		for article in article_list[0:1]:
			summary = self.decoder.DecodeOne(article)
			summary_list.append(summary)

		summary = " ".join(summary_list)

		print("ehre4")

		return (actual_abstract,summary)


## Test Case

# source = codecs.open("my.tex", "r","utf-8")
# print( summarize_with_string(source.read()))
# source.close()
#summarize('comm_magazine_FDMIMO_overview.tex')

