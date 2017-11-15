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

    def load(self, fname):
        super(Lda, self).load(fname)
        self.model = models.LdaMulticore.load(fname)

    def get_embeddings(self, corpus, norm=False):
        """returns embedding matrix
        Arguments:

            corpus: corpus that yields one document at a time in gensim
                sprse (list of 2-tuples) format.

            norm: boolean (default=False). Whether to normalize the embeddings.
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
