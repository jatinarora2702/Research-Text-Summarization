from SimpleXMLRPCServer import SimpleXMLRPCServer
from SimpleXMLRPCServer import SimpleXMLRPCRequestHandler

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
		os.system('python2 ../../../Codes/parser.py '+filename)

		# Run Extractive Summary
		os.system('python2 ../../../Codes/plot_accuracy_graph.py '+filename)



		# path - /home/bt1/13CS10060/snlp16/Test_Data/data/summary/lex-rank/100

		base_dir = os.path.dirname(os.path.realpath(__file__))
		base_dir = os.path.dirname(base_dir)
		base_dir = os.path.dirname(base_dir)
		base_dir = os.path.dirname(base_dir)



		article_unprocessed = ""
		with open(base_dir + "/Test_Data/data/summary/lex-rank/100/"+filename) as f:
			article_unprocessed = f.read()

		article = pre_example_process(get_doc_from_tokens(article_to_tokens(article_unprocessed)))

		#print(article)

		summary = self.decoder.DecodeOne(article)

		#summary = ""

		## Post process model summary if required

		actual_summary_file = codecs.open(base_dir + "/Test_Data/data/model/"+filename, 'r', 'utf-8')

		actual_summary = actual_summary_file.read()

		return (actual_summary,summary)


## Test Case

# source = codecs.open("my.tex", "r","utf-8")
# print( summarize_with_string(source.read()))
# source.close()
#summarize('comm_magazine_FDMIMO_overview.tex')

