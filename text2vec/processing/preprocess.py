"""methods for processing text.
"""
import re
import string
from typing import Callable, List, Tuple, Union
from collections import Counter
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

STOP_WORDS = stopwords.words('english')
STOP_WORDS_PUNCT_REGEX = re.compile(
    '^' + '$|^'.join(STOP_WORDS + [re.escape(s) for s in string.punctuation]) + '$',
    re.IGNORECASE)
DIGIT_REGEX = re.compile(r'\d+')

def preprocess_one(text: str,
                   tokenize: Callable[[str], List[str]] = None,
                   pipeline: List[dict] = None) -> List[str]:
    """tokenizes a document and conducts other preprocessing.

    Arguments:

        text: string representing document to be tokenized.

        tokenize: function. tokenizer that splits the text into a list of
            tokens. Takes the text string as the first input argument

        pipeline: list of dicts. List of preprocessing functions to call
            on the tokenized text. Each element in the list should be a dict with
            two keys: "returns" (either "bool" or "str") and "function".
            The "returns" key indicates whether the preprocesser returns a
            transformed string (e.g. stems a word) or returns a Boolean
            indicating whether a token should be discarded (True) or kept (False).

            If None, uses the following preprocessors by default:

                [
                    {"returns": "str", "function": PorterStemmer().stem},
                    {"returns": "bool", "function": rm_stop_words_punct}
                ]

            - `rm_stop_words_punct` removes English stop words (from
            nltk.corpus.stopwords) and punctuation (from string.punctuation).
            - `stem` stems tokens using nltk.stem.porter.PorterStemmer. The text
             will be processed in the order that the preprocessors are given in
             this pipeline.

            If you do not wish to use any preprocessors, pass an empty list: [].

    Returns:

        tokens: list of str. List of strings representing the processed words.

    Example:

        >>> from nltk.tokenize import RegexpTokenizer
        >>> from nltk.stem.porter import PorterStemmer
        >>> from utils import preprocess, rm_stop_words_punct
        >>> text = 'this is a test example. (some text in parentheses!)'
        >>> tokenize = RegexpTokenizer(r'\w+').tokenize  # NOTE: this is the default if you pass tokenize=None.
        >>> pipeline = [  # NOTE: this is the default if you pass pipeline=None.
                {"returns": "str", "function": PorterStemmer().stem},
                {"returns": "bool", "function": rm_stop_words_punct}
            ]
        >>> preprocess(text, tokenize, pipeline)
        ['thi', 'test', 'exampl', 'text', 'parenthes']
    """
    tokens = []
    if tokenize is None:
        tokenize = RegexpTokenizer(r'\w+').tokenize
    if pipeline is None:
        pipeline = [
            {"returns": "bool", "function": rm_stop_words_punct},
            {"returns": "str", "function": PorterStemmer().stem}
        ]
    if text is not None and len(text) > 0:
        tokens = tokenize(text)
        for processor in pipeline:
            if processor['returns'] == 'bool':
                tokens = [token for token in tokens if not processor['function'](token)]
            elif processor['returns'] == 'str':
                tokens = [processor['function'](token) for token in tokens]
            else:
                raise ValueError('{0} return type not implemented'.format(processor['returns']))
    return tokens


def rm_stop_words_punct(token: str) -> bool:
    """returns true if token is an nltk English stop words or a piece of
    punctuation.

    Arguments:

        token: string.

    Returns:

        bool: True if token is an nltk English stop word or a piece of
            punctuation; False otherwise.
    """
    # tokens = [token for token in tokens if token not in STOP_WORDS]
    return bool(re.search(STOP_WORDS_PUNCT_REGEX, token))


def rm_digits(token: str) -> bool:
    """returns true if token contains one or more digits.

    Arguments:

        token: string.
    
    Returns:

        bool: True if token contains one or more digits, False otherwise.
    """
    return bool(re.search(DIGIT_REGEX, token))


def rm_min_chars(min_len: int) -> Callable[[str], bool]:
    """returns Callable that takes a token as the only argument and returns
    True if the token length (number of characters) is less than min_len; False
    otherwise.

    Arguments:

        min_len: int. Minimum length.

    Returns:

        func: Callable[[str], bool].

    Example::

        >>> func = rm_min_chars(10)
        >>> func('longwordisgood')
        False
        >>> func('tooshort')
        True
    """
    func = lambda token: len(token) < min_len
    return func


def tokens2bow(tokens: List[str]) -> List[Tuple[Union[int, str], int]]:
    """converts a list of tokens to 2-tuple bag of words format.

    Arguments:

        tokens: list of str or int. List of tokens or token_ids.

    Returns:

        list of 2-tuples. Each element in the list is a 2-tuple, where the first
            element of the tuple represents the original token/token_id and the
            second element represents the number of times that token/token_id
            appears in the list of input tokens.

    Example:

        >>> tokens2bow(['my', 'name', 'is', 'X', '.', 'your', 'name', 'is', 'Y', '.'])
        [('my', 1), ('your', 1), ('Y', 1), ('X', 1), ('is', 2), ('.', 2), ('name', 2)]
        >>> tokens2bow([1, 2, 1, 3, 4, 5, 6, 2, 8, 9, 1])
        [(1, 3), (2, 2), (3, 1), (4, 1), (5, 1), (6, 1), (8, 1), (9, 1)]
    """
    return list(Counter(tokens).items())
