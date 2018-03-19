import nltk
from nltk.corpus import movie_reviews
from nltk.classify import NaiveBayesClassifier
import nltk.classify.util as util
from nltk.collocations import BigramCollocationFinder as BCF
from nltk.metrics import BigramAssocMeasures
import itertools


def features(words):
	scoreF = BigramAssocMeasures.chi_sq

	#bigram count
	n = 430

	bigrams = BCF.from_words(words).nbest(scoreF, n)

	return dict([word,True] for word in itertools.chain(words, bigrams))

def main():

	pid = movie_reviews.fileids('neg')
	nid = movie_reviews.fileids('pos')

	prev = [(features(movie_reviews.words(fileids = id)), 'positive') for id in pid]
	nrev = [(features(movie_reviews.words(fileids = id)), 'negative') for id in nid]

	ncutoff = int(len(nrev)*3/4)
	pcutoff = int(len(prev)*3/4)

	train_set = nrev[:ncutoff] + prev[:pcutoff]
	test_set = nrev[ncutoff:] + prev[pcutoff:]


	# NaiveBayesClassifier
	classifier = NaiveBayesClassifier.train(train_set)

	# Accuracy
	print ("Accuracy is : ", util.accuracy(classifier, test_set) * 100)

if (__name__ == "__main__"):
	main()
