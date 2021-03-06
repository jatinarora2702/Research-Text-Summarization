import gensim
import sys
import os
import codecs
import json
from preprocess import preprocess
import nltk.data
import multiprocessing

import pickle
from collections import OrderedDict


basepath = "../../../Test_Data/Output_Dict/"


tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


dict_paras = {}
dict_sentences= {}

def get_embedding(filename, trained_model):
	file_path=basepath+filename+'.txt'
	print("here4", file_path)
	train = codecs.open(file_path,'r','utf-8')
	papers=[]

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
	
	paper_no =0

	li_vecs = []
	for paper in papers:
		paper_no += 1
		#print(itr_no,"art",paper_no,paper['id'])
		paper['sum']=paper['sum'].encode('utf-8')
		for key in paper['info']:
			paper_data=""
			for item in paper['info'][key]:
				if isinstance(item,str):
					paper_data+=item+" "
				elif isinstance(item,bytes):
					paper_data+=item+" "
				elif isinstance(item,dict):
					for innerKey in item:
						for innerItem in item[innerKey]:
							if (isinstance(innerItem,str)):
								paper_data+=innerItem+" "
							elif (isinstance(innerItem,bytes)):
								paper_data+=innerItem+" "
							elif isinstance(innerItem,dict):
								for in_innerKey in innerItem:
									for in_innerItem in innerItem[in_innerKey]:
										if (isinstance(in_innerItem,str)):
											paper_data+=in_innerItem+" "
										elif (isinstance(in_innerItem,bytes)):
											paper_data+=in_innerItem+" "
			curr_vec = trained_model.infer_vector(preprocess(paper_data), alpha=0.1, min_alpha=0.0001, steps=5)
			li_vecs.append(curr_vec)
		print("End ", paper['sum'])
		temp_file = open("vec_pic.pkl",'wb')

		result = (li_vecs, paper['sum'])

		pickle.dump(result, temp_file, protocol = 2)

		return "vec_pic.pkl"           