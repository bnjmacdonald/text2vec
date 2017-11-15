import numpy as np

# def filter_dictionary(dictionary, no_below, no_above):
#     orig_token_sample = [dictionary.id2token[i] for i in range(100)]
#     dictionary_filt = copy.deepcopy(dictionary)
#     dictionary_filt.filter_extremes(no_below=no_below, no_above=no_above)
#     word_ids = [dictionary.token2id[w] for w in dictionary_filt.values()]  # word ids to keep
#     dictionary.filter_tokens(good_ids=word_ids)
#     assert len(word_ids) == len(dictionary_filt)
#     assert len(word_ids) == len(dictionary)
#     dictionary[0]  # required for gensim to build id2token dictionary.
#     assert all([orig_token_sample[i] == dictionary.id2token[i] for i in range(100)])
#     return dictionary_filt

def normalize(vecs):
    norms = np.linalg.norm(vecs, ord=2, axis=1, keepdims=True)
    vecs = np.divide(vecs, norms)
    return vecs

def sliced_stream(rows, streamer):
    """returns a slice of rows from a file that is streamed from disk.

    streamer must be a generator (or other iterable) that yields one line
    per iter.

    Example::

        stream_slice(rows=[50, 100, 187], streamer=corpus.stream_documents())
    """
    # extracts slice.
    sliced = []
    for i, l in enumerate(streamer):
        if i in rows:
            sliced.append(l)
    # orders slice by rows.
    result = []
    for i in rows:
        result.append(sliced[i])
    return result
