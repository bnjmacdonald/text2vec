"""unit tests."""

# import sys
# sys.path.append('.')
import os
import time
import json
import settings
from corpora import Corpus

def test_processor():
    time0 = time.time()
    test_fname = os.path.join(settings.DATA_DIR, 'test_docs.json')
    out_path = os.path.join(settings.OUTPUT_DIR, 'debug')
    with open(test_fname, 'r') as f:
        data = json.load(f)

    ids, documents = zip(*[(k, v['body']) for k, v in data.items()])
    corpus = Corpus(path=out_path, verbose=1)
    corpus.mk_corpus(
        documents=documents,
        ids=ids,
        tokenizer=None,
        stem=False,
        rm_stop_words=False,
        rm_punct=True,
    )
    time1 = time.time()
    corpus.mk_dictionary()
    time2 = time.time()
    corpus.mk_corpus_bow()
    time3 = time.time()
    corpus.get_sizes()
    time4 = time.time()
    corpus.load_dictionary()

    print('Time to construct corpus: {0}'.format(time1 - time0))
    print('Time to construct dictionary: {0}'.format(time2 - time1))
    print('Time to construct bow corpus: {0}'.format(time3 - time2))
    print('Time to get sizes: {0}'.format(time4 - time3))
    return corpus

if __name__ == '__main__':
    corpus = test_processor()