"""unit tests."""

# import sys
# sys.path.append('.')
import os
import time
import json
import settings
import processor

def test_processor():
    time0 = time.time()
    test_fname = os.path.join(settings.DATA_DIR, 'test_docs.json')
    out_path = os.path.join(settings.OUTPUT_DIR, 'debug')
    with open(test_fname, 'r') as f:
        data = json.load(f)
    ids, documents = zip(*[(k, v['body']) for k, v in data.items()])
    corpus = processor.CorpusProcessor(verbose=1)
    corpus.mk_corpus(
        documents=documents,
        ids=ids,
        tokenizer=None,
        stem=False,
        rm_stop_words=False,
        rm_punct=True,
        dict_filter_kws={}
    )
    time1 = time.time()
    corpus.save(
        out_path=out_path,
        export_documents=True,
        # config_kws=config_dict
    )
    time2 = time.time()
    corpus.load(input_path=out_path, import_documents=True)
    time3 = time.time()
    print('Time to construct corpus: {0}'.format(time1 - time0))
    print('Time to save corpus: {0}'.format(time2 - time1))
    print('Time to load corpus: {0}'.format(time3 - time2))
    return corpus

if __name__ == '__main__':
    corpus = test_processor()