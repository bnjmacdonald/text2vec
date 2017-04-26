from setuptools import setup

setup(
    name='text2vec',
    version='0.1',
    description='Implementations of various methods for converting a chunk of text into a dense vector.',
    url='https://github.com/bnjmacdonald/text2vec',
    author='Bobbie NJ Macdonald',
    author_email='bnjmacdonald@gmail.com',
    license='MIT',
    packages=['text2vec'],
    install_requires=[
        'gensim',
        'nltk',
        'numpy',
      ],
    zip_safe=False
)