from setuptools import setup

setup(
    name='text2vec',
    version='0.11',
    description='Implementations of various methods for converting a chunk of text into a dense vector.',
    url='https://github.com/bnjmacdonald/text2vec',
    author='Bobbie NJ Macdonald',
    author_email='bnjmacdonald@gmail.com',
    license='MIT',
    packages=[
        'text2vec',
        'text2vec.corpora',
        'text2vec.models',
        'text2vec.processing',
        'text2vec.inspect',
        'text2vec.inspect.assign_dim_names'
    ],
    install_requires=[
        'gensim',
        'nltk==3.2.1',
        'numpy',
        'sklearn',
        'joblib',
        'typing'
      ],
    zip_safe=False
)