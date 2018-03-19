import nltk
from nltk.corpus import movie_reviews
from nltk.classify import NaiveBayesClassifier
import nltk.classify.util as util


def features(words):
	return dict([word,True] for word in words)

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
