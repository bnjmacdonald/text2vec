"""unit tests."""

import sys
sys.path.append('.')
import os
import json
import settings
import processor

def test_processor():
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
    corpus.save(
        out_path=out_path,
        export_documents=True,
        # config_kws=config_dict
    )
    corpus.load(input_path=out_path, import_documents=True)
    return corpus

if __name__ == '__main__':
    corpus = test_processor()