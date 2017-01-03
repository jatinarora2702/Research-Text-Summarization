import gensim
import sys
import os
import codecs
import json
from preprocess import preprocess
import nltk.data
import multiprocessing


tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

file_path = "../Dataset/all-parsed-papers-category.txt"


dict_paras = {}
dict_sentences= {}

def init():
	train = codecs.open(file_path,'r','utf-8')
	papers=[]

	# model_s_DM = gensim.models.doc2vec.Doc2Vec.load('Models/sentences_DM.doc2vec')
	# model_s_DBOW = gensim.models.doc2vec.Doc2Vec.load('Models/sentences_DBOW.doc2vec')
	# model_s_DBOW = gensim.models.doc2vec.Doc2Vec.load('Models/sentences_DBOW.doc2vec')
	# model_s_DBOW = gensim.models.doc2vec.Doc2Vec.load('Models/sentences_DBOW.doc2vec')

	for line in train:
		line = line.replace("###FORMULA###","||FORMULA||")
		line = line.replace("###TABLE###","||TABLE||")
		line = line.replace("###FIGURE###","||FIGURE||")

		map=line.split('\t')
		paper=dict()
		paper['id']=map[0]
		paper['name']=map[1]
		try:
			paper['info']=json.loads(map[2])
		except:
			continue
		paper['sum']=map[3]
		if (len(paper['sum'])>=10):
			papers.append(paper)
	print("Paper ", len(papers))
	for paper in papers:
		print(paper['id'])
		paper['sum']=paper['sum'].encode('utf-8')
		paper_data = ""
		for key in paper['info']:
			paper_data_abs=""
			for item in paper['info'][key]:
				if isinstance(item,str):
					paper_data+=item+" "
					paper_data_abs+=item+" "

				elif isinstance(item,bytes):
					paper_data+=item+" "
					paper_data_abs+=item+" "
				elif isinstance(item,dict):
					for innerKey in item:
						for innerItem in item[innerKey]:
							if (isinstance(innerItem,str)):
								paper_data+=innerItem+" "
								paper_data_abs+=innerItem+" "

							elif (isinstance(innerItem,bytes)):
								paper_data+=innerItem+" "
								paper_data_abs+=innerItem+" "

							elif isinstance(innerItem,dict):
								for in_innerKey in innerItem:
									for in_innerItem in innerItem[in_innerKey]:
										if (isinstance(in_innerItem,str)):
											paper_data+=in_innerItem+" "
											paper_data_abs+=in_innerItem+" "

										elif (isinstance(in_innerItem,bytes)):
											paper_data+=in_innerItem+" "
											paper_data_abs+=in_innerItem+" "
			para_label = str(paper['id']) + '_' + str(key)

			if paper['id'] not in dict_paras:
				dict_paras[paper['id']] = []

			dict_paras[paper['id']].append(para_label)

		lines = tokenizer.tokenize(paper_data)

		for uid, line in enumerate(lines):
			sentence_label = str(paper['id']) + '_' + str(uid)

			if paper['id'] not in dict_sentences:
				dict_sentences[paper['id']] = []

			dict_sentences[paper['id']].append(sentence_label)
	return papers
	print("Initialization done")

def get_embedding(article_fname, para = True, algo = "DM"):
	paper_vector = []
	if para == False:

		if algo == "DM":
			model = gensim.models.doc2vec.Doc2Vec.load('Models_New/sentences_DM.doc2vec')

		elif algo == "DBOW":
			model = gensim.models.doc2vec.Doc2Vec.load('Models_New/sentences_DBOW.doc2vec')

		else:
			print("error")
			return []
		
		sentence_labels = dict_sentences[article_fname]



		for label in sentence_labels:
			origin = model.docvecs[label]

			paper_vector.append(origin)


	elif para == True:

		if algo == "DM":
			model = gensim.models.doc2vec.Doc2Vec.load('Models_New/paragraph_DM.doc2vec')

		elif algo == "DBOW":
			model = gensim.models.doc2vec.Doc2Vec.load('Models_New/paragraph_DBOW.doc2vec')

		else:
			print("error")
			return []
		
		para_labels = dict_paras[article_fname]



		for label in para_labels:
			origin = model.docvecs[label]

			paper_vector.append(origin)


	return paper_vector

init()

print(len(dict_paras))

print(dict_paras['1603.04918.txt'])
print(len(get_embedding('1603.04918.txt')))

print(get_embedding('1603.04918.txt'))


