from bert_bindings.transformer_model_vectorizer import FlairTransformerEmbedding
from w2v_tf_idf_bindings.w2v_tfi_df_vectorizer import Word2Vec_TF_IDF_Vectorizer
from w2v_bindings.w2v_vectorizer import Word2VecVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
bert_data ={ 'embedding_transformer' : joblib.load("bert_bindings\\bert_embedding_transformer.pkl"),
'cosine_similarity_mat' : joblib.load('bert_bindings\\cosine_similarity_bert.pkl')}

tf_idf_data ={ 'embedding_transformer' : joblib.load("tf_idf_bindings\\tfidf_embedding_transformer.pkl"),
'cosine_similarity_mat' : joblib.load('tf_idf_bindings\\cosine_similarity_tfidf.pkl')}

w2v_tf_idf_data ={ 'embedding_transformer' : joblib.load("w2v_tf_idf_bindings\\w2v_tf_idf_embedding_transformer.pkl"),
'cosine_similarity_mat' : joblib.load('w2v_tf_idf_bindings\\cosine_similarity_w2v_tf_idf.pkl')}

w2v_data ={ 'embedding_transformer' : joblib.load("w2v_bindings\\w2v_embedding_transformer.pkl"),
'cosine_similarity_mat' : joblib.load('w2v_bindings\\cosine_similarity_w2v.pkl')}