import numpy as np

def normalize(vecs):
    norms = np.linalg.norm(vecs, ord=2, axis=1, keepdims=True)
    vecs = np.divide(vecs, norms)
    return vecs
