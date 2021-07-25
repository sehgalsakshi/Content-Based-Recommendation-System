from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import joblib
google_model = joblib.load('google_model.pkl')
class Word2VecVectorizer(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
      # Creating a list for storing the vectors (description into vectors)
      word_embeddings = []

      # Reading the each book description 
      for line in X:
        avgword2vec = None
        count = 0
        for word in line.split():
            if word in google_model.wv.vocab:
                count += 1
                if avgword2vec is None:
                    avgword2vec = google_model[word]
                else:
                    avgword2vec = avgword2vec + google_model[word]
                
        if avgword2vec is not None:
            avgword2vec = avgword2vec / count
        if avgword2vec is None:
          print(line)
        else:
          word_embeddings.append(avgword2vec)#.reshape(-1,1))
      word_embeddings = np.asarray(word_embeddings)
      return word_embeddings