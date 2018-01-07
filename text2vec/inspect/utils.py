import numpy as np

def dim_top_docs(dim: int, embeddings: np.ndarray, topn: int = 10) -> np.array:
    """retrieves the documents that load highest on an embedding dimension.

    Arguments:

        dim: int. Dimension from which to sample.

        embeddings: np.ndarray (shape: n_obs, n_dimensions).

        topn: int. Number of top documents to retrieve.

    Returns:

        top_doc_indices: np.array (shape: topn, ). np.array of row indices of
            the documents that load highest on the dimension.
    """
    inv_rankings = embeddings[:, dim].argsort()
    top_doc_indices = inv_rankings[(embeddings.shape[0] - topn)::][::-1]
    # embeddings[max_indices, dim][::-1]
    return top_doc_indices


def docs_top_dim(embeddings: np.ndarray, method: str = 'positive') -> np.array:
    """retrieves top embedding for each document.

    Arguments:

        embeddings: np.ndarray (shape: n_obs, n_dimensions).

        method: str. Method for ranking documents. Available options::

            positive: retrieves top document based on largest value of a
                document's embedding vector.

            negative: retrieves top document based on smallest value of a
                document's embedding vector.

            absolute: retrieves top document based on largest absolute value of
                a document's embedding vector. This corresponds to a measure of
                which embedding a document loads most strongly on, regardless of
                directionality.

            deviation: retrieves top documents based on largest absolute
                deviation between the two largest embeddings of a document's
                embedding vector.

            percentile: retrieves top documents after converting each the
                embedding matrix to column-based percentile rankings. Then
                retrieves documents with highest percentile rank.

    Returns:

        np.array. np.array of ints, where each element corresponds to the top
            dimension for that document.
    """
    if method == 'positive':
        return embeddings.argmax(axis=1)
    elif method == 'negative':
        return embeddings.argmin(axis=1)
    elif method == 'absolute':
        return np.absolute(embeddings).argmax(axis=1)
    elif method == 'deviation':
        raise NotImplementedError
    elif method == 'percentile':
        raise NotImplementedError
    else:
        raise RuntimeError('unknown method {0}'.format(method))


def sample_dim_docs(dim: int, embeddings: np.ndarray, **kwargs) -> np.array:
    """samples documents from a single dimension.

    Arguments:

        dim: int. Dimension from which to sample.

        embeddings: np.ndarray (shape: n_obs, n_dimensions).

        **kwargs: Keyword arguments to pass to np.random.choice.

    Returns:

        sample: np.array (shape: kwargs['size'], ). np.array of row indices of
            sampled documents.

            Example::

                np.array([45921, 23211, 550302, 92832, 119321])
    """
    top_dims = docs_top_dim(embeddings, method='positive')
    indices = np.array(range(0, embeddings.shape[0]))
    # [doc for i, doc in enumerate(documents) if top_dims[i] == dim]
    sample = np.random.choice(indices[top_dims == dim], **kwargs)
    return sample
