import numpy as np
from scipy.spatial import distance


import nltk

import gensim


def space_distance(B, ui):
	val = ui
	for bj in B:
		dot = np.dot(bj, ui)
		proj = np.multiply(bj, dot)
		val = val - proj
	return np.linalg.norm(val)




def summarize(sentences, L):
	'''
		sentences: list of Sentences, containing text, rep, id, order
	'''
	summary = []
	B = []

	words_n = 0

	centroid = sentences[0]['vector']

	N = len(sentences)

	for idx,s in enumerate(sentences[1:]):
		ui = s['vector']
		centroid = np.add(centroid, ui)

	centroid = np.divide(centroid, float(len(sentences)))

	maxdist = distance.euclidean(centroid,sentences[0]['vector'])
	p = 0

	for idx,s in enumerate(sentences[1:]):
		ui = s['vector']
		if (maxdist < distance.euclidean(ui, centroid)): p=idx


	maxdist = distance.euclidean(sentences[p]['vector'],sentences[0]['vector'])
	q = 0

	for s in sentences[1:]:
		ui = s['vector']
		if maxdist < distance.euclidean(ui, sentences[p]['vector']):q = idx 

	summary.append(sentences[p])
	summary.append(sentences[q])

	

	words_n = len(sentences[p]['words']) + len(sentences[q]['words'])

	uq = sentences[q]['vector']

	B.append(np.divide(uq, np.linalg.norm(uq)))

	del sentences[p]
	del sentences[q]

	for ii in range(N-2):
		r = 0
		maxv = space_distance(B, sentences[0]['vector'])
		for idx, s in enumerate(sentences[1:]):
			ui = s['vector']
			if maxv < space_distance(B, ui):r = idx 

		if(words_n <= L):
			summary.append(sentences[r])
			ur = sentences[r]['vector']
			B.append(np.divide(ur, np.linalg.norm(ur)))
			words_n += len(sentences[r]['words'])
			del sentences[r]

	return summary




def test_algo():
	text ='''The Norse-American medal was struck at the Philadelphia Mint in 1925, pursuant to an act of the United States Congress.
	It was issued for the 100th anniversary of the voyage of the ship Restauration,	bringing early Norwegian immigrants to the United States.
	Minnesota Congressman Ole Juulson Kvale, a Norse-American, wanted a commemorative for the centennial celebrations of the Restauration journey.
	Rebuffed by the Treasury Department when he sought the issuance of a special coin, he instead settled for a medal.
	Sculpted by Buffalo nickel designer James Earle Fraser, the medals recognize those immigrants' Viking heritage, depicting a warrior on the obverse and a vessel on the reverse.
	They also recall the early Viking explorations of North America.
	Once authorized by Congress, they were produced in various metals and sizes, for the most part prior to the celebrations near Minneapolis in June 1925.
	Only 53 were issued in gold, and they are rare and valuable today; those struck in silver or bronze have appreciated much less in value.
	They are sometimes collected as part of the U.S. commemorative coin series.'''
	import pickle
	try:
		sents = pickle.load(open("test.json","rb"))
	except:

		model = gensim.models.Word2Vec.load_word2vec_format('btp/codes/embeddings/google_news_300.bin', binary=True)  # C binary format
		sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
		from nltk.tokenize import TreebankWordTokenizer
		word_detector = TreebankWordTokenizer()

		sentences = sent_detector.tokenize(text.strip())

		sents = []

		for idx,s in enumerate(sentences):
			d = {}
			d['sent'] = s
			d['words'] = word_detector.tokenize(s.strip())
			d['ord'] = idx


			v = np.zeros(300)

			for word in d['words']:
				try:
					v = v + model[word.lower()]
				except:
					pass
			d['vector'] = np.divide(v , len(sentences))

			sents.append(d)

	
		pickle.dump(sents, open('test.json', "wb"))



	#print(sents)


	summary = summarize(sents, 60)
	from operator import itemgetter
	sorteds = sorted(summary, key=itemgetter('ord'))

	for s in sorteds:
		print(s['sent'])


test_algo()









