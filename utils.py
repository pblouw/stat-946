import nltk
import collections
import string
import os
import re
import numpy as np

stopwords = nltk.corpus.stopwords.words('english')
tokenizer = nltk.load('tokenizers/punkt/english.pickle')
min_len = 4

def docstream(size=None):
    '''Stream lists of documents from wikipedia'''
    if size is None:
        count = np.inf
    else:
        count = size
    doc_break = '</doc>'
    for root, dirs, files in os.walk(os.getcwd()+'/data/wikidump/'):
        for fname in files: 
            if 'wiki' in fname:
                dlist=[]
                with open(os.path.join(root, fname), 'r') as f:
                    text = f.read().split(doc_break)                   
                    for item in text:
                        doc = re.sub("<.*>", "", item)
                        try:
                            doc = doc.decode('unicode_escape')
                            doc = doc.encode('ascii','ignore')
                        except UnicodeDecodeError:
                            print 'Unicode Error Warning'
                            continue
                        dlist.append(doc)
                yield dlist
                count -= 1
                if count <= 0:
                    raise StopIteration()

                
def countwords(doc):
    counts = collections.Counter()
    sen_list = tokenizer.tokenize(doc)
    sen_list = [s.replace('\n', ' ') for s in sen_list]
    sen_list = [s.translate(None, string.punctuation) for s in sen_list]
    sen_list = [s.translate(None, '1234567890') for s in sen_list]
    sen_list = [s for s in sen_list if len(s.split()) >= min_len]
    sen_list = [[w.lower() for w in s.split()] for s in sen_list]
    for sen in sen_list:
        counts.update(sen)
    return counts