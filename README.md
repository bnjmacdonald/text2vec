# README

This project implements various methods for converting a chunk of text into a dense vector (e.g. LDA, LSI, seq2seq, doc2vec, ...) and provides convenient utilities for pre-processing text so that it can be fed to any of these models.

NOTE: this repository is under active development. The current version only implements two simple bag of words models (LDA and LSI) and provides utilities for managing/building corpora and dictionaries. Next step is to refactor deep learning code from my `behavioral styles` project and add it to this repo.

## Usage

coming soon...

## Terminology

TODO: update this terminology to reflect recent changes.

- `documents`: array of strings, dicts, or objects where each element represents a single unprocessed (raw) document. The documents are not tokenized or pre-processed. Example: the text of a news article as a single string.
  - `doc`: a single unprocessed document (e.g. "hi, my name is...").
- `doc_arrays`: list of lists containing documents that have been preprocessed but not yet converted into a specific `bow` of `seq` format.
  - `doc_array`: a single preprocessed document.
  - NOTE: `doc_arrays` sometimes consists of a list of 2-tuples, where the first element in each tuple is the document _id and the second element is the `doc_array`.
  - `doc_array_padded`: a right-padded `doc_array`.
- `corpus`: matrix or list of 2-tuples representing a preprocessed corpus that is ready to be fed to a `text2vec.fit` method.
  - `row`: a single preprocessed document that has been converted into `bow` or `seq` format.
  - `embeddings`: 2d array representing lower-dimensional document embeddings. Each row is an embedding vector for a single document.
    - `doc_embed`: embedding vector for a single document.
- `dictionary': dict of id->word pairs. Common words: '<UNK>' (unknown word), '<EOD>' (end of document).
  - `token_id`: an integer representing the id for a single word.
  - `token`: a string representing a single word.
- `ids`: array of ids that uniquely identify each document in the corpus. The ids must be sorted in the same order as the first axis of documents and corpus.
  - `_id`: document id.
- `config`: JSON dict containing information about how the corpus was constructed and/or how the model was estimated (e.g. whether stop words were removed, etc).

Corpus build types:

- `bow`: each document is converted to a "bag of words", consisting of a count of the number of times each word in the corpus appears in the document. By default, bag of words are computed from unigrams.
- `seq`: each document is converted to a sequence of tokens representing each word in the document. This sequence of tokens retains word order. The sequence can either be right-padded or non-padded. If right-padded, a padding (e.g. zeros) as appended to the end of each sequence so that all documents in the corpus have the same number of tokens (making the corpus rectangular).
