
"""manually assigns a name/label to each embedding dimension.

Todos:

    TODO: implement simple web app that provides an interface for naming
        embedding dimensions.
"""

import json
from typing import List, Callable, Dict
from pprint import pprint
import numpy as np

from text2vec.inspect.utils import dim_top_docs, sample_dim_docs


def assign_manual(outpath: str,
                  dims: List[int],
                  document_ids: np.array,
                  ids2tokens: Callable[[List[int]], List[str]],
                  embeddings: np.ndarray,
                  token_embeddings: list,
                  document_getter: Callable[[list], List[dict]]) -> Dict[int, Dict[str, str]]:
    """provides a simple command-line interface for assigning embedding
    dimension names manually.

    You are asked to assign a name to each embedding dimension. To decide on a
    dimension's name, you can choose the following options in the command-line
    interface:

        [n] assign a name to this dimension \
        [v] view names of other dimensions \
        [x] skip to next dimension \
        [s] sample documents from dimension \
        [d] view top documents for dimension \
        [t] view top tokens for dimension \
        [q] quit

    The names are stored in a dict of dim -> name mappings (e.g. {0:
    "healthcare", 1: "police violence"}). This dict is saved to `outpath` after
    every name is assigned, so that you can quit at any time and not lose your
    progress.

    Notes:

        For each embedding dimension, prints out:

            - a sample of documents that rank highly on that dimension;
            - a sample of all documents for which that dimension has the largest
                absolute value;
            - a list of the highest loading tokens (words, usually) for that embedding
                dimension;
            - a list of the words with the largest distance between their loading on
                this dimension and loading on other dimensions.

    Arguments:

        outpath: str. Path to directory where `dim_names.json` file should be
            saved. If `dim_names.json` already exists, these existing names
            are loaded and will not be overwritten.

        dims: list of int. List of dimensions to name.

        document_ids: np.array (shape: n_documents, ). Array of document _ids
            that uniquely identify each document.

        ids2tokens: Callable. Callable that takes a list of token _ids as first
            argument and returns list of str tokens corresponding to these ids.

        embeddings: np.ndarray (shape: n_documents, n_dimensions). np.ndarray
            of document embeddings. Each row is an embedding vector for a
            document. Order of rows must correspond to order of document _ids
            in document_ids.

        token_embeddings: np.ndarray (shape: n_tokens, n_dimensions). np.ndarray
            of token embeddings. Each row is a single token (e.g. word,
            character) embedding vector.

        document_getter: Callable. Callable that receives a list of document_ids
            and retrieves the full document for each _id.

    Returns:

        names: Dict[int, Dict[str, str]]. Dictionary of names for
            each dimension. Each key points to a dict containing a name,
            short_name, and description for each dimension.

            Example::

                {
                    0: {
                        "short_name": "pharmaceuticals",
                        "name": "pharmaceutical prices",
                        "description": "Speeches emphasize concerns over high 
                            pharmaceutical prices."
                    },
                    1: {
                        "short_name": ...,
                        "name": ...,
                        "description": ...
                    },
                    ...
                }
    """
    try:
        with open(outpath, 'r') as f:
            names = {int(dim): name_data for dim, name_data in json.load(f).items()}
    except FileNotFoundError:
        names = {}
    actions = {
        'n': lambda dim, topn: assign_name(dim, names=names, outpath=outpath),
        'v': lambda dim, topn: names,
        's': lambda dim, topn: document_getter(
            document_ids[sample_dim_docs(dim=dim, embeddings=embeddings, size=topn, replace=False)]
        ),
        'd': lambda dim, topn: document_getter(
            document_ids[dim_top_docs(dim=dim, embeddings=embeddings, topn=topn)]
        ),
        't': lambda dim, topn: ids2tokens(
            dim_top_docs(dim=dim, embeddings=token_embeddings, topn=topn)
        ),
    }
    action_msg = lambda dim, name: '\n' + '-'*20 + '\nASSIGNING NAME TO DIMENSION {0}.\
        \nExisting dimension name: {1} \
        \nWhat do you want to do? \
        \n\t[n] assign a name to this dimension \
        \n\t[v] view names of other dimensions \
        \n\t[x] skip to next dimension \
        \n\t[s] sample documents from dimension \
        \n\t[d] view top documents for dimension \
        \n\t[t] view top tokens for dimension \
        \n\t[q] quit \
        \ncommand: '.format(dim, name)
    try:
        for dim in dims:
            name = None
            existing_name = names[dim] if dim in names else None
            while name is None:
                action = input(action_msg(dim, existing_name))
                if action == 'x':
                    print('passing over dimension {0}'.format(dim))
                    break
                else:
                    name = _handle_action(actions, action, dim)
    except KeyboardInterrupt:
        if len(names) > 0:
            print('Assigned names: ')
            pprint(names)
            print('Saved assigned names to {0}'.format(outpath))
    return names


def _handle_action(actions: dict, action: str, dim: int) -> str:
    """handles user action."""
    topn = None
    topn_default = 10
    try:
        if action == 'q':
            raise KeyboardInterrupt
        elif action in ['s', 'd', 't']:
            while not isinstance(topn, int):
                topn = input('Number to view (hit enter for default: {0}): '.format(topn_default))
                if len(topn.strip()) == 0:
                    topn = topn_default
                else:
                    try:
                        topn = int(topn.strip())
                    except ValueError:
                        print('{0} is not a valid integer'.format(topn))
                        topn = None
        result = actions[action](dim, topn)
        if action == 'n':
            return result  # name of dimension
        elif action == 'v':
            print('\n')
            pprint(result)
        elif action == 's':
            for doc in result:
                print('\n')
                pprint(doc)
                input('press any key to view next: ')
        elif action == 'd':
            for doc in result:
                print('\n')
                pprint(doc)
                input('press any key to view next: ')
        elif action == 't':
            print('\n')
            print(result)
    except KeyError:
        print('\n{0} not recognized.'.format(action))
    return None


def assign_name(dim: int, names: dict, outpath: str) -> str:
    """assigns name to a dimension."""
    confirmed = False
    while not confirmed:
        name = input('name for dimension {0} ([m]ain menu): '.format(dim))
        name = name.strip().lower()
        if name == 'm':
            return None
        if _is_valid_name(name):
            confirmed = _confirm_name(name)
    names[dim] = {'name': name}
    with open(outpath, 'w') as f:
        json.dump(names, f)
        print('{0} name for dimension {1} saved to {2}'.format(name, dim, outpath))
    return name


def _is_valid_name(name: str) -> bool:
    """checks if user input name is valid."""
    if len(name) == 0:
        return False
    return True


def _confirm_name(name: str) -> bool:
    """asks user for confirmation of dimension name."""
    msg = 'This dim will be named "{0}". Do you wish to continue (y/n)? '.format(name)
    confirmed = input(msg)
    confirmed = confirmed.lower().strip() if confirmed else None
    while confirmed not in ['y', 'n']:
        print('"{0}" not recognized. Please type "y" to confirm or "n" to enter a new name.'.format(confirmed))
        confirmed = input(msg)
        confirmed = confirmed.lower().strip() if confirmed else None
    assert confirmed in ['y', 'n'], 'something went wrong. "{0}" is not "y" or "n"'.format(confirmed)
    if confirmed == 'y':
        return True
    return False
