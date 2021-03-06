import os
import numpy as np
from gensim import models, matutils
from text2vec.utils import normalize
from text2vec.text2vec import Text2Vec


class Lda(Text2Vec):
    """Defines the LDA model.

    NOTE: max_seqlen has no effect on lda models.
    """

    def __str__(self):
        return 'lda'

    def _train(self, corpus, **kwargs):
        """trains the LDA model.

        Thin wrapper to `gensim.models.LDAMulticore`.

        Arguments:

            corpus (corpus): corpus that yields one document at a time in gensim
                sparse (list of 2-tuples) format.

            **kwargs: keyword arguments to pass to `gensim.models.LDAMulticore`.
        """
        # corpus_bow = matutils.Sparse2Corpus(corpus_csr, documents_columns=False)
        model = models.LdaMulticore(corpus, **kwargs)
        self.model = model
        # Todo: fix this "pd.DataFrame constructor not properly called" error.
        # self.embeddings = self.get_embeddings(corpus, norm=False)

    def save(self, path):
        super(Lda, self).save(path)
        self.model.save(os.path.join(path, self._model_fname))

    def load(self, path):
        super(Lda, self).load(path)
        self.model = models.LdaMulticore.load(os.path.join(path, 'model'))

    def embed(self, corpus, norm=False):
        """returns embedding matrix

        Arguments:

            corpus: corpus that yields one document at a time in gensim
                sprse (list of 2-tuples) format.

            norm: boolean (default=False). Whether to normalize the embeddings.

        Todos:

            TODO: can this be made faster?
        """
        # corpus_bow = matutils.Sparse2Corpus(corpus_csr, documents_columns=False)
        orig_embeddings = self.model[corpus]
        orig_embeddings_csr = matutils.corpus2csc(orig_embeddings, num_terms=self.model.num_topics).transpose()
        embeddings = orig_embeddings_csr.todense()
        if norm:
            embeddings = normalize(embeddings)
        return embeddings

    def infer_docvecs(self, corpus):
        """

        Arguments:

            corpus: ...

        Todos:

            TODO: doc_ids (i.e. result index) are currently meaningless.
        """
        if isinstance(corpus, np.ndarray) or matutils.ismatrix(corpus):
            corpus = matutils.Dense2Corpus(corpus, documents_columns=False)
        docvecs_infer = self.model[corpus]
        docvecs_infer_csr = matutils.corpus2csc(docvecs_infer, num_terms=self.model.num_topics).transpose()
        docvecs_infer = docvecs_infer_csr.todense()
        # docvecs_infer = np.vstack(docvecs_infer)
        return docvecs_infer
    
    def token_embed(self, token_ids):
        """retrieves token embeddings (n_tokens X n_topics matrix).
        """
        token_embeddings = []
        for token_id in token_ids:
            arr = np.zeros((self.model.num_topics,))
            term_topics = self.model.get_term_topics(token_id, minimum_probability=0.0)
            for topic, loading in term_topics:
                arr[topic] = loading
            token_embeddings.append(arr)
        token_embeddings = np.vstack(token_embeddings)
        assert token_embeddings.shape[0] == len(token_ids)
        return token_embeddings
