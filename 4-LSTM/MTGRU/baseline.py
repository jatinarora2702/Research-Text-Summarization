# -*- coding: utf8 -*-

from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals
print("here1")
from sumy.parsers.html import HtmlParser
from sumy.parsers.plaintext import PlaintextParser
print("here2")
from sumy.nlp.tokenizers import Tokenizer
print("here3")
from sumy.summarizers.lex_rank import LexRankSummarizer as Summarizer

print("here4")
# from sumy.summarizers.lex_rank_embed import LexRankEmbedSummarizer as Summarizer2
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

# from sumy.embeddings.embed_model import setpath

import os
import sys
import codecs


LANGUAGE = "english"


def runsumy(method, num, ip_file_path, op_file_path):
    parser = PlaintextParser.from_file(ip_file_path, Tokenizer(LANGUAGE))
    stemmer = Stemmer(LANGUAGE)

    summarizer = Summarizer(stemmer)
    summarizer.stop_words = get_stop_words(LANGUAGE)

    # f = codecs.open(op_file_path, 'w', 'utf-8')
    s = ""
    for word in summarizer(parser.document, int(num)):
    	s += word._text.encode('utf-8').decode('utf-8')
    	# print(word._text.encode('utf-8'), file=f) # not outputing in the designated file
    return s


base_dir = "../../../Test_Data/Output_Dict/"

def extractive_intro(filename,method="lex-rank",num="1"):
	print("here base")
	# base_dir = sys.argv[1]
	# method = sys.argv[2]
	# num = sys.argv[3]
	# empath = sys.argv[4]
	ip_dir = base_dir + "/data/input/"
	op_dir = base_dir + "/data/summary/"


	# setpath(empath)

	# ip_files = os.listdir(ip_dir)

	ip_file = codecs.open(ip_dir+filename)

	# for ip_file in ip_files:
	ip_file_path = ip_dir + ip_file
	op_file_path = op_dir + method + "/" + num
	if not os.path.exists(op_file_path):
		os.makedirs(op_file_path)
	op_file_path = op_file_path+ "/" + filename
	
	# command = "sumy " + method + " --length=" + num + " --file=\'" + ip_file_path + "\' > \'" + op_file_path + "\'"
	# os.system(command)
	summary_lines = runsumy(method, num, ip_file_path, op_file_path)
	print(ip_file)

	return summary_lines

	# print("Method : " + method + ", Num : " + num)


if __name__ == "__main__":
	main()
