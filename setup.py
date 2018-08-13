#!/usr/bin/env python

from setuptools import setup
from setuptools import find_packages

setup(name='UKP_FactoidQA_on_Wikidata',
      version='0.0.1',
      description='UKP factoid question answering system over WikiData',
      author='Daniil Sorokin',
      author_email='sorokin@ukp.informatik.tu-darmstadt.de',
      url='ukp.tu-darmstadt.de/ukp-home/',
      packages=find_packages(), requires=['numpy', 'nltk', 'tqdm', 'SPARQLWrapper', 'click', 'keras', 'sklearn',
                                          'torch', 'flask'])
