import codecs
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize,StanfordTokenizer
from nltk.stem.porter import *
from stemming.porter2 import stem

stopWordPath="english.txt"

stop = []

def isfloat(x):
    try:
        a = float(x)
    except ValueError:
        return False
    else:
        return True

def isint(x):
    try:
        a = float(x)
        b = int(a)
    except ValueError:
        return False
    else:
        return a == b

def getStopWords():
	slist = []

	swords = codecs.open(stopWordPath,"r","utf-8")
	for words in swords:
		word = words.strip("\r\n ")
		slist.append(word)
	return slist
	

def getnltkstops():
	from nltk.corpus import stopwords
	return stopwords.words('english')

def stemWord(word):
	return word
	#return word


def tokenize(s):
	tokens = []
	for t in sent_tokenize(s):
		tokens.extend(word_tokenize(t))
	return tokens

def stanford_tokenize(s):
	return StanfordTokenizer().tokenize(s)


def preprocess_old(text):
	_tokens = tokenize(text)

	stripstring = ',.?;\':"[]{}()! '
	# tokens = [token.strip(stripstring).lower() for token in _tokens if token.strip(stripstring).lower() not in stop and not token.strip(stripstring).lower().startswith('#$#') and len(token.strip(stripstring).lower()) > 0 ]
	tokens = [token.strip(stripstring).lower() for token in _tokens if len(token.strip(stripstring).lower()) > 0 ]

	for idx, token in enumerate(tokens):
		if isfloat(token) or isint(token):
			tokens[idx] = "||NUMBERTOKEN||"

	return tokens


# def init():
# 	global stop
# 	stop = getStopWords()
# 	punc = list(".!?'\"$;:,()[]{}")
# 	stop.append("'s")
# 	stop.extend(punc)

def preprocess(text,_type="T"):
	tokens = preprocess_old(text)
	return tokens