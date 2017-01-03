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


itr_no = 0

class LabeledLineSentence(object):
	def __init__(self, filename):
		self.filename = filename
	def __iter__(self):

		global itr_no
		train = codecs.open(self.filename,'r','utf-8')
		papers=[]

		paper_no = 0

		itr_no += 1

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

			paper_no += 1
			print(itr_no,"art",paper_no,paper['id'])
			paper['sum']=paper['sum'].encode('utf-8')

			paper_data=""

			for key in paper['info']:
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

			lines = tokenizer.tokenize(paper_data)
			for uid, line_1 in enumerate(lines):
				yield gensim.models.doc2vec.LabeledSentence(words=preprocess(line_1), tags=[str(paper['id']) + '_' + str(uid)])

		train = codecs.open(self.filename,'r','utf-8')
		papers=[]

		abs_no = 0
		for line in train:
			line = line.replace("###FORMULA###","||FORMULA||")
			line = line.replace("###TABLE###","||TABLE||")
			line = line.replace("###FIGURE###","||FIGURE||")

			abs_no += 1

			print(itr_no,"abs",abs_no,paper['id'])

			map=line.split('\t')
			paper_id = map[0]
			summary=map[3]
			print(paper_id)

			lines = tokenizer.tokenize(summary)
			for uid, line_1 in enumerate(lines):
				yield gensim.models.doc2vec.LabeledSentence(words=preprocess(line_1), tags=['ABS_'+str(paper_id)+'_'+str(uid)])


class LabeledParagraph(object):
	def __init__(self, filename):
		self.filename = filename
	def __iter__(self):

		global itr_no
		
		train = codecs.open(self.filename,'r','utf-8')
		papers=[]

		itr_no += 1

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

		paper_no =0
		for paper in papers:
			paper_no += 1
			print(itr_no,"art",paper_no,paper['id'])
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

				yield gensim.models.doc2vec.LabeledSentence(words=preprocess(paper_data), tags=[str(paper['id']) + '_' + str(key)])

		train = codecs.open(self.filename,'r','utf-8')

		abs_no = 0

		for line in train:
			line = line.replace("###FORMULA###","||FORMULA||")
			line = line.replace("###TABLE###","||TABLE||")
			line = line.replace("###FIGURE###","||FIGURE||")
			abs_no += 1

			
			map=line.split('\t')
			paper_id = map[0]
			print(itr_no,"abs",abs_no,paper['id'])
			summary=map[3]
			yield gensim.models.doc2vec.LabeledSentence(words=preprocess(summary), tags=['ABS_'+str(paper_id)])




class LabeledAbstractSentence(object):
	def __init__(self, filename):
		self.filename = filename
	def __iter__(self):
		train = codecs.open(self.filename,'r','utf-8')
		papers=[]
		for line in train:
			line = line.replace("###FORMULA###","||FORMULA||")
			line = line.replace("###TABLE###","||TABLE||")
			line = line.replace("###FIGURE###","||FIGURE||")

			map=line.split('\t')
			paper_id = map[0]
			summary=map[3]
			print(paper_id)

			lines = tokenizer.tokenize(summary)
			for uid, line in enumerate(lines):
				yield gensim.models.doc2vec.LabeledSentence(words=preprocess(summary), tags=['ABS_'+str(paper_id)+'_'+str(uid)])

class LabeledAbstractParagraph(object):
	def __init__(self, filename):
		self.filename = filename
	def __iter__(self):
		train = codecs.open(self.filename,'r','utf-8')
		
		


		for line in train:
			line = line.replace("###FORMULA###","||FORMULA||")
			line = line.replace("###TABLE###","||TABLE||")
			line = line.replace("###FIGURE###","||FIGURE||")

			
			map=line.split('\t')
			paper_id = map[0]
			print(paper_id)
			summary=map[3]
			yield gensim.models.doc2vec.LabeledSentence(words=preprocess(summary), tags=['ABS_'+str(paper_id)])

def  train_sentences(algo = "DM"):
	sentences = LabeledLineSentence(file_path)

	if algo == "DM":
		model_DM = gensim.models.doc2vec.Doc2Vec(sentences, size = 300, window = 8, min_count=1, workers=multiprocessing.cpu_count(), iter = 10,  dm = 1, negative=10)
		print("Model sentences_DM Trained")
		model_DM.save("Models_New/sentences_DM.doc2vec")
		print("Model sentences_DM saved")

	elif algo == "DBOW":
		model_DBOW = gensim.models.doc2vec.Doc2Vec(sentences, size = 300, window = 8, min_count=1, workers=multiprocessing.cpu_count(), iter = 10,  dm = 0, negative=10)
		print("Model sentences_DBOW Trained")
		model_DBOW.save("Models_New/sentences_DBOW.doc2vec")
		print("Model sentences_DBOW saved")

def  train_paragraph(algo = "DM"):
	paragraphs = LabeledParagraph(file_path)

	if algo == "DM":
		model_DM = gensim.models.doc2vec.Doc2Vec(paragraphs, size = 400, window = 10, min_count=1, workers=multiprocessing.cpu_count(), iter = 10,  dm = 1, negative=10)
		print("Model paragraphs_DM Trained")
		model_DM.save("Models_New/paragraph_DM.doc2vec")
		print("Model paragraphs_DM saved")

	elif algo == "DBOW":
		model_DBOW = gensim.models.doc2vec.Doc2Vec(paragraphs, size = 400, window = 10, min_count=1, workers=multiprocessing.cpu_count(), iter = 10,  dm = 0, negative=10)
		print("Model paragraph_DBOW Trained")
		model_DBOW.save("Models_New/paragraph_DBOW.doc2vec")
		print("Model paragraphs_DBOW saved")



def train_abstract(para = False, algo = "DM"):
	if para == True:
		abs_paras = LabeledAbstractParagraph(file_path)

		if algo == "DM":
			fname = "Models_New/paragraph_"+algo+".doc2vec"
			model_DM = gensim.models.doc2vec.Doc2Vec.load(fname)
			print("abs para DM loaded")
			model_DM.train(abs_paras)
			print("abs para DM trained")
			model_DM.save(fname)
			print("abs para DM saved")
		elif algo == "DBOW":
			fname = "Models_New/paragraph_"+algo+".doc2vec"
			model_DBOW = gensim.models.doc2vec.Doc2Vec.load(fname)
			print("abs para DBOW loaded")
			model_DBOW.train(abs_paras)
			print("abs para DBOW trained")
			model_DBOW.save(fname)
			print("abs para DBOW saved")




	elif para == False:
		abs_sentences = LabeledAbstractSentence(file_path)

		if algo == "DM":
			fname = "Models_New/sentences_"+algo+".doc2vec"
			model_DM = gensim.models.doc2vec.Doc2Vec.load(fname)
			print("abs sentences DM loaded")
			model_DM.train(abs_sentences)
			print("abs sentences DM trained")
			model_DM.save(fname)
			print("abs sentences DM saved")
		elif algo == "DBOW":
			fname = "Models_New/sentences_"+algo+".doc2vec"
			model_DBOW = gensim.models.doc2vec.Doc2Vec.load(fname)
			print("abs sentences DBOW saved")
			model_DBOW.train(abs_sentences)
			print("abs sentences DBOW saved")
			model_DBOW.save(fname)
			print("abs sentences DBOW saved")



def main():

	abs_or_article = sys.argv[1]

	s_or_p = sys.argv[2]

	DM_or_DBOW = sys.argv[3]


	if abs_or_article == "abs":
		if s_or_p == "s":
			if DM_or_DBOW == "DM":
				train_abstract(False,"DM")
			elif DM_or_DBOW == "DBOW":
				train_abstract(False,"DBOW")

		elif s_or_p == "p":
			if DM_or_DBOW == "DM":
				train_abstract(True,"DM")

			elif DM_or_DBOW == "DBOW":
				train_abstract(True,"DBOW")

		else:
			print("Invalid Option")


	elif abs_or_article == "art":
	
		if s_or_p == "s":
			if DM_or_DBOW == "DM":
				train_sentences("DM")
			elif DM_or_DBOW == "DBOW":
				train_sentences("DBOW")

		elif s_or_p == "p":
			if DM_or_DBOW == "DM":
				train_paragraph("DM")

			elif DM_or_DBOW == "DBOW":
				train_paragraph("DBOW")

		else:
			print("Invalid Option")


if __name__ == "__main__":main()