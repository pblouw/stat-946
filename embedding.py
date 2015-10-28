import operator
import string
import nltk
import numpy as np
import collections
import multiprocessing
from utils import docstream, countwords

class EmbeddingModel(object):
    """
    Base class for models that define word embeddings.
    """        
    stopwords = nltk.corpus.stopwords.words('english')
    tokenizer = nltk.load('tokenizers/punkt/english.pickle')
    min_len = 4 # minimum sentence length

    @staticmethod
    def softmax(z):
        return np.exp(z) / np.sum(np.exp(z), axis=0)

    def get_sents(self, doc):
        sen_list = self.tokenizer.tokenize(doc)
        sen_list = [s.replace('\n', ' ') for s in sen_list]
        sen_list = [s.translate(None, string.punctuation) for s in sen_list]
        sen_list = [s.translate(None, '1234567890') for s in sen_list]
        sen_list = [s for s in sen_list if len(s.split()) >= self.min_len]
        sen_list = [[w.lower() for w in s.split()] for s in sen_list]
        sen_list = [[w for w in s if w in self.vocab] for s in sen_list]
        return sen_list
        
    def get_context(self, pos, sen):
        context = []
        for i in range(self.win_size):
            if pos+i+1 < len(sen):
                context.append(sen[pos+i+1])
            if pos-i-1 >= 0:
                context.append(sen[pos-i-1])
        return list(set(context))
    
    def get_binvec(self, context):
        binvec = np.zeros(len(self.vocab))
        for word in context:
            binvec += self.get_onehot(word)
        return binvec 

    def get_onehot(self, word):
        index = self.word_indices[word]
        onehot = np.zeros(len(self.vocab))
        onehot[index] = 1
        return onehot

    def get_synonyms(self, word):
        probe = self.word_vecs[self.word_indices[word], :]
        self.rank_words(np.dot(self.word_vecs, probe))

    def indices_to_words(self, v):
        indices = np.where(v!=0)[0]
        words = []
        for key, val in self.word_indices.iteritems():
            if val in inds:
                words.append(key)
        return words    

    def rank_words(self, scores):
        rank = zip(range(len(self.vocab)), scores)
        rank = sorted(rank, key=operator.itemgetter(1), reverse=True)
        top_words = [(self.vocab[x[0]],x[1]) for x in rank[:10]]
        print ''
        for word in top_words[:5]:
            print word[0], word[1]

    def data(self, size, model):
        counter = 0 
        for doclist in docstream():
            for doc in doclist:
                if counter >= size:
                    raise StopIteration()
                counter += 1
                sen_list = self.get_sents(doc)
                for sen in sen_list:
                    sen = [w for w in sen if w not in self.stopwords]
                    if len(sen) < 4:
                        continue
                    xs = np.zeros((len(self.vocab),len(sen)))
                    ys = np.zeros((len(self.vocab),len(sen)))
                    if model == 'cbow':
                        for _ in range(len(sen)):
                            context = self.get_context(_,sen)
                            xs[:,_] = self.get_binvec(context)
                            ys[:,_] = self.get_onehot(sen[_])
                        yield xs, ys
                    elif model == 'skipgram':
                        for _ in range(len(sen)):
                            context = self.get_context(_,sen)
                            xs[:,_] = self.get_onehot(sen[_])
                            ys[:,_] = self.get_binvec(context)
                        yield xs, ys
            