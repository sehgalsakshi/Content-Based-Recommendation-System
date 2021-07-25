from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
google_model = joblib.load('google_model.pkl')
tfidf = joblib.load('w2v_tf_idf_bindings\\tf_idf_vectorizer.pkl')
tfidf_list = dict(zip(tfidf.get_feature_names(), list(tfidf.idf_)))
tfidf_feature = tfidf.get_feature_names() # tfidf words/col-names

class Word2Vec_TF_IDF_Vectorizer(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
      # Storing the TFIDF Word2Vec embeddings
      tfidf_vectors = []; 
      line = 0
      # for each book description
      for desc in X: 
        # Word vectors are of zero length (Used 300 dimensions)
          sent_vec = np.zeros(300) 
          # num of words with a valid vector in the book description
          weight_sum =0; 
          # for each word in the book description
          for word in desc.split(): 
              if word in google_model.wv.vocab and word in tfidf_feature:
                  vec = google_model.wv[word]
                  tf_idf = tfidf_list[word] * (desc.count(word) / len(desc))
                  sent_vec += (vec * tf_idf)
                  weight_sum += tf_idf
          if weight_sum != 0:
              sent_vec /= weight_sum
          tfidf_vectors.append(sent_vec)
          line += 1
      # Creating a list for storing the vectors (description into vectors)
      word_embeddings = np.asarray(tfidf_vectors)
      return word_embeddings