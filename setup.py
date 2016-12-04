#!/usr/bin/env python

from setuptools import setup
from setuptools import find_packages

setup(name='UKPWikiDataQA',
      version='0.0.1',
      description='UKP Question answering over WikiData',
      author='Daniil Sorokin',
      author_email='sorokin@ukp.informatik.tu-darmstadt.de',
      url='ukp.tu-darmstadt.de/ukp-home/',
      packages=find_packages(), requires=['numpy', 'nltk', 'tqdm', 'SPARQLWrapper', 'click', 'keras'])
