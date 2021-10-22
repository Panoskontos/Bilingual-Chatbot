from nltk.stem.porter import PorterStemmer
import nltk
import numpy as np
nltk.download('punkt')


stemmer = PorterStemmer()


def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def stem(word):
    return stemmer.stem(word.lower())


def bag_of_words(tokenised_sentence, all_words):
    tokenised_sentence = [stem(w) for w in tokenised_sentence]

    # create arrays [0,0,0,0,0]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for index, w in enumerate(all_words):
        if w in tokenised_sentence:
            bag[index] = 1.0

    return bag
    # return array with found words [0,0,1,0,1]