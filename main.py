# Importing necessary libraries
import pandas as pd
from bert_bindings.transformer_model_vectorizer import FlairTransformerEmbedding
from w2v_tf_idf_bindings.w2v_tfi_df_vectorizer import Word2Vec_TF_IDF_Vectorizer
from w2v_bindings.w2v_vectorizer import Word2VecVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from recommendation_system import recommend_books
from load_generated_embeddings import bert_data, w2v_data, w2v_tf_idf_data, tf_idf_data
import json

# Reading the data
data_dict = bert_data
output = recommend_books(data_dict['embedding_transformer'], data_dict['cosine_similarity_mat'],  title='the fountainhead')

print(json.dumps(output, indent = 4))