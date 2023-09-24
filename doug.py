"""
Module to perform Vector database load
"""

from aimakerspace.text_utils import TextFileLoader, CharacterTextSplitter
from aimakerspace.vectordatabase import VectorDatabase

def dg_read_file(filename):
    text_loader = TextFileLoader(filename)
    return text_loader.load_documents()

def dg_split_file(document):
    text_splitter = CharacterTextSplitter()
    return text_splitter.split_texts(document)

