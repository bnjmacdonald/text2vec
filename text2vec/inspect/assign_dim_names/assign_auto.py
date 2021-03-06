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

from typing import Dict, List
import numpy as np

def assign_auto(model: object,
                method: str = 'top',
                topn: int = 10,
                unique: bool = False,
                verbosity: int = 0) -> Dict[int, Dict[str, str]]:
    """assigns dimension names automatically by extracting the most common
    token in the topic and using that token as the dimension name.

    Arguments:

        model: object. Text2Vec model instance.

        method: str (default: 'top'). Method to use for assigning the dimension
            name. Available options:

                top: most frequently occuring word is assigned as the dimension
                    name.
                
                random: a random word among the topn is assigned as the dimension
                    name.
        
        topn: int (default: 10). Only used if method == 'random'.

        unique: bool (default: False). If True, ensures that all dimensions are
            given a unique name.

    Returns:

        names: Dict[int, Dict[str, str]]. Dictionary of names for
            each dimension. Each key points to a dict containing a name and
            short_name for each dimension. The short_name is the most frequently
            occuring token. The name is a comma-separated list of the top five
            most frequently occuring tokens.

            Example::

                {
                    0: {
                        "short_name": "pharm",
                        "name": "pharm, price, health, cost, hosp",
                    },
                    1: {
                        "short_name": ...,
                        "name": ...,
                        "description": ...
                    },
                    ...
                }

    Todos:

        TODO: this method is currently only implemented for LDA models.
            Implement this method for other models as well.

        TODO: implement use of `unique` argument.
    """
    if str(model) != 'lda':
        raise NotImplementedError
    if unique:
        raise NotImplementedError
    names = {}
    for i in range(model.config['num_topics']):
        # retrieves top words and loadings.
        top_words, loadings = zip(*model.model.show_topic(topicid=i, topn=topn))
        if verbosity > 0:
            print('\n------------------\nTopic %d:' % (i))
            print('Top words:', top_words)
            print('Loading:', loadings)
        names[i] = {
            "name": ', '.join(top_words[0:topn]),
            "short_name": _assign_dim_name(top_words, method, unique, verbosity=verbosity)
        }
    return names


def _assign_dim_name(words: List[str], method: str, unique: bool, verbosity: int = 0) -> str:
    """Helper method that invokes the appropriate method to retrieve a dimension
    name. Returns a dimension name.
    """
    if unique:
        raise NotImplementedError
    if method == 'top':
        return words[0]
    elif method == 'random':
        return np.random.choice(words)
    else:
        raise RuntimeError('{0} method not recognized'.format(method))
