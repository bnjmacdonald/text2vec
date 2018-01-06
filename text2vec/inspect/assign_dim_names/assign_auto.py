"""assigns names to each embedding dimension automatically.

Notes:

    NOTE: really this is a classification problem, where I seek to
        predict a single word conditional on a vector of word loadings. We could
        do this in a supervised by if we are constantly re-estimating models and
        manually assigning names (which would provide a training set). But how
        would we do this in an unsupervised way? One solution would be to simply
        take a weighted sum of the top words and then find the word that is
        closest to this average. An even simpler approach would be to grab the
        top word.

Todos:

    TODO: imeplement method that assign dimension name based on most "standout"
        token, rather than simply the most common token. (e.g. tf-idf?)
"""

def assign_auto(model: object, topn: int, verbosity: int = 0):
    """assigns dimension names automatically by extracting the most common token
    in the topic and using that token as the dimension name.

    Todos:
        
        TODO: this is currently only implemented for LDA models. Implement this
            method for other models as well.
    """
    if str(model) != 'lda':
        raise NotImplementedError
    embed_names = {}
    for i in range(model.config['num_topics']):
        top_words, loadings = zip(*model.model.show_topic(topicid=i, topn=topn))  # top words and loadings.
        if verbosity > 0:
            print('\n------------------\nTopic %d:' % (i))
            print('Top words:', top_words)
            print('Loading:', loadings)
        embed_names[i] = top_words[0]
    return embed_names
