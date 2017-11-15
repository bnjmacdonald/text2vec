from gensim import matutils

def corpus_bow2array(corpus):
    """converts a bag-of-words corpus to a dense corpus.

    Todos:

        TODO: check if this implementation is faster than::

            X = matutils.corpus2csc(self.stream_corpus_bow()).transpose()
    """
    X = matutils.corpus2csc(corpus).transpose()
    # num_words = len(self.dictionary)
    # vecs = []
    # for i, doc in enumerate(self.stream_corpus_bow()):
    #     a = np.zeros((num_words,))
    #     word_ids, counts = zip(*doc)
    #     a[list(word_ids)] = counts
    #     # for word_id, count in doc:
    #     #     a[word_id] = count
    #     vecs.append(a)
    #     if i > 0 and i % 10000 == 0:
    #         assert a.sum() == sum(counts)
    #         print('Converted {0} documents so far'.format(i), end='\r')
    # print()
    # X = np.vstack(vecs)
    return X
