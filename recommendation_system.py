import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import joblib

df = pd.read_csv('good_reads_data.csv')

#adding embeddings to dataframe
df['bert_vectors'] = joblib.load('bert_bindings\\bert_vectors_list.pkl')
df['w2v_vectors'] = joblib.load('w2v_bindings\\w2v_vectors_list.pkl')
df['tf_idf_vectors'] = joblib.load('tf_idf_bindings\\tfidf_vectors_list.pkl')
df['w2v_tf_idf_vectors'] = joblib.load('w2v_tf_idf_bindings\\w2v_tf_idf_vectors_list.pkl')

def get_text_vector(text, column_transformer):
  text_np_array = np.asarray([text])
  text_np_array = text_np_array.reshape(-1,1)
  text_np_array = column_transformer.transform(text_np_array)
  return text_np_array

def find_most_similar_vector_list(input_vector, vector_col):
  cosine_similarity_list = []
  for row in df.itertuples():
    cosine_similarity_value = cosine_similarity(input_vector, [getattr(row, vector_col)])
    cosine_similarity_list.append(cosine_similarity_value[0][0])
  return cosine_similarity_list

def create_book_dict(idx_score):
  books_dict = df.iloc[idx_score[0]].to_dict()
  print(idx_score)
  books_dict['score']=float(idx_score[1])
  books_dict['id']=float(idx_score[0])
  keys_to_retain = ['id','Desc', 'title', 'author', 'genre', 'image_link', 'rating', 'score']
  return dict((key,value) for key, value in books_dict.items() if key in keys_to_retain)

def get_recommendations_list(cosine_list, num_books, idx = None, filter_by_rating=True, thresh_score=.85):
    cosine_list = sorted(enumerate(cosine_list), key=lambda i: i[1], reverse=True)
    if filter_by_rating:
        cosine_thresh_list = cosine_list.copy()
        cosine_thresh_list = [(book, df.iloc[book[0]].to_dict()['rating']) for i, book in enumerate(cosine_thresh_list)\
                                if (book[1]>=thresh_score and len(cosine_thresh_list[:i])<=(2*num_books))\
                                or len(cosine_thresh_list[:i])<=num_books]
        cosine_thresh_list = (sorted(cosine_thresh_list, key=lambda i: i[1], reverse=True))[:num_books+1]
        cosine_list = [book[0] for book in cosine_thresh_list]
    else:
        cosine_list = (sorted(enumerate(cosine_list), key=lambda i: i[1], reverse=True))[:num_books+1]
    cosine_list = [ create_book_dict(idx_score) for idx_score in cosine_list if not idx or int(idx_score[0]) != int(idx)]
    return cosine_list

def recommend_books_from_desc(desc, embedding_transformer, vector_col, num_books=5):
  desc_vector = get_text_vector(desc, embedding_transformer)
  return find_most_similar_vector_list(desc_vector, vector_col)

def recommend_books(embedding_transformer, cosine_similarity_matrix, title=None, desc=None, vector_col='bert_vectors', num_books=5,
                               filter_by_rating=True, thresh_score=.85):
  if title:
    recommended_books = {}
    for count, idx in enumerate(df[df.title.str.lower() == title.lower()].index):
      cosine_list = [i[0] for i in cosine_similarity_matrix[idx].reshape(-1,1)]
      recommended_books[idx] = get_recommendations_list(cosine_list, num_books, idx=idx , filter_by_rating=filter_by_rating, thresh_score=thresh_score)
    return recommended_books
  else:
    cosine_list = recommend_books_from_desc(desc, embedding_transformer, vector_col, num_books=num_books)
    return get_recommendations_list(cosine_list, num_books, filter_by_rating=filter_by_rating, thresh_score=thresh_score)
  