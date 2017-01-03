import gensim
import sys
import os
import codecs
import json
from preprocess import preprocess
import nltk.data
import multiprocessing

from collections import OrderedDict


tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

file_path = "all-parsed-papers-category.txt"


dict_paras = {}
dict_sentences= {}

def init(para=True, algo = "DM"):
	train = codecs.open(file_path,'r','utf-8')
	papers=[]

	if para == False:

		if algo == "DM":
			model = gensim.models.doc2vec.Doc2Vec.load('Models/sentences_DM.doc2vec')

		elif algo == "DBOW":
			model = gensim.models.doc2vec.Doc2Vec.load('Models/sentences_DBOW.doc2vec')

	elif para == True:

		if algo == "DM":
			model = gensim.models.doc2vec.Doc2Vec.load('Models/paragraph_DM.doc2vec')

		elif algo == "DBOW":
			model = gensim.models.doc2vec.Doc2Vec.load('Models/paragraph_DBOW.doc2vec')
	print("Model Initialization done")

	for line in train:
		line = line.replace("###FORMULA###","||FORMULA||")
		line = line.replace("###TABLE###","||TABLE||")
		line = line.replace("###FIGURE###","||FIGURE||")

		map=line.split('\t')
		paper=dict()
		paper['id']=map[0]
		paper['name']=map[1]
		try:
			paper['info']=json.loads(map[2],object_pairs_hook=OrderedDict)
		except:
			continue
		paper['sum']=map[3]
		if (len(paper['sum'])>=10):
			papers.append(paper)
	print("Paper ", len(papers))
	for paper in papers:
		li_labels = []
		paper['sum']=paper['sum'].encode('utf-8')
		paper_data = ""
		for key in paper['info']:
			para_label = str(paper['id']) + '_' + str(key)

			# if paper['id'] not in dict_paras:
			# 	dict_paras[paper['id']] = []


			origin = model.docvecs[para_label]

			# paper_vector.append(origin)

			li_labels.append(origin)

		print(paper['id'], paper['name'])

		yield (paper['id'],paper['sum'],li_labels, paper['name'])

		'''
		lines = tokenizer.tokenize(paper_data)

		for uid, line in enumerate(lines):
			sentence_label = str(paper['id']) + '_' + str(uid)

			if paper['id'] not in dict_sentences:
				dict_sentences[paper['id']] = []

			dict_sentences[paper['id']].append(sentence_label)
		'''
	# return papers
	print("Initialization done")

def get_embedding(article_fname, para = True, algo = "DM"):
	paper_vector = []
	if para == False:

		if algo == "DM":
			model = gensim.models.doc2vec.Doc2Vec.load('Models/sentences_DM.doc2vec')

		elif algo == "DBOW":
			model = gensim.models.doc2vec.Doc2Vec.load('Models/sentences_DBOW.doc2vec')

		else:
			print("error")
			return []
		
		sentence_labels = dict_sentences[article_fname]



		for label in sentence_labels:
			origin = model.docvecs[label]

			paper_vector.append(origin)


	elif para == True:

		if algo == "DM":
			model = gensim.models.doc2vec.Doc2Vec.load('Models/paragraph_DM.doc2vec')

		elif algo == "DBOW":
			model = gensim.models.doc2vec.Doc2Vec.load('Models/paragraph_DBOW.doc2vec')

		else:
			print("error")
			return []
		
		# para_labels = dict_paras[article_fname]



		# for label in para_labels:

	return paper_vector



#print(get_embedding('1603.04918.txt'))
