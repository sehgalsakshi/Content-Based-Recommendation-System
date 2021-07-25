import torch
from flair.data import Sentence
from flair.embeddings import TransformerDocumentEmbeddings
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class FlairTransformerEmbedding(TransformerMixin, BaseEstimator):

    def __init__(self, model_name='bert-base-uncased', batch_size=None, layers=None):
        # From https://lvngd.com/blog/spacy-word-vectors-as-features-in-scikit-learn/
        # For pickling reason you should not load models in __init__
        self.model_name = model_name
        self.model_kw_args = {'batch_size': batch_size, 'layers': layers}
        self.model_kw_args = {k: v for k, v in self.model_kw_args.items()
                              if v is not None}
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        model = TransformerDocumentEmbeddings(
                self.model_name, fine_tune=False,
                **self.model_kw_args)

        sentences = [Sentence(text) for text in X]
        embedded = model.embed(sentences)
        embedded = [e.get_embedding().reshape(1, -1) for e in embedded]
        return np.array(torch.cat(embedded).cpu())