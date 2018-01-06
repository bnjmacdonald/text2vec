import numpy as np

def dim_top_docs(dim: int, embeddings: np.ndarray, topn: int = 10) -> np.array:
    """retrieves the documents that load highest on an embedding dimension.
    """
    inv_rankings = embeddings[:,dim].argsort()
    top_doc_indices = inv_rankings[(embeddings.shape[0] - topn)::][::-1]
    # embeddings[max_indices, dim][::-1]
    return top_doc_indices


def docs_top_dim(embeddings: np.ndarray, method: str = 'positive') -> np.array:
    """retrieves top embedding for each document.
        
        positive: retrieves top document based on largest value of a document's
            embedding vector.

        negative: retrieves top document based on smallest value of a document's
            embedding vector.

        absolute: retrieves top document based on largest absolute value of a
            document's embedding vector. This corresponds to a measure of
            which embedding a document loads most strongly on, regardless of
            directionality.

        deviation: retrieves top documents based on largest absolute deviation
            between the two largest embeddings of a document's embedding vector.
        
        percentile: retrieves top documents after converting each the embedding matrix
            to column-based percentile rankings. Then retrieves documents with
            highest percentile rank for 
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
    top_dims = docs_top_dim(embeddings, method='positive')
    indices = np.array(range(0, embeddings.shape[0]))
    # [doc for i, doc in enumerate(documents) if top_dims[i] == dim]
    sample = np.random.choice(indices[top_dims == dim], **kwargs)
    return sample
