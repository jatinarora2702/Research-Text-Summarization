import gensim

embedding_path = None
model = None

def setpath(path):
	embedding_path = path

def model():
	if model is not None:
		return model

	if path is None:
		raise ValueError("Embedding path must be set")

	model = gensim.models.Word2Vec.load_word2vec_format(embedding_path, binary=True)

	return model
