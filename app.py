import __main__
from flask import Flask,request,jsonify
from flask_cors import CORS,cross_origin
from bert_bindings.transformer_model_vectorizer import FlairTransformerEmbedding
from w2v_tf_idf_bindings.w2v_tfi_df_vectorizer import Word2Vec_TF_IDF_Vectorizer
from w2v_bindings.w2v_vectorizer import Word2VecVectorizer
from recommendation_system import recommend_books
from load_generated_embeddings import bert_data, w2v_data, w2v_tf_idf_data, tf_idf_data
import json
from preprocess import make_lower_case, remove_html, remove_punctuation, remove_stop_words, _removeNonAscii
app = Flask(__name__)
__main__.FlairTransformerEmbedding = FlairTransformerEmbedding
__main__.Word2Vec_TF_IDF_Vectorizer = Word2Vec_TF_IDF_Vectorizer
__main__.Word2VecVectorizer = Word2VecVectorizer
app.config.from_object(__name__)
cors = CORS(app)
def get_data_dict(embedding_type):
    if embedding_type == 'bert':
        return bert_data
    elif embedding_type == 'w2v':
        return w2v_data
    elif embedding_type == 'w2v_tf_idf':
        return w2v_tf_idf_data
    else:
        return tf_idf_data

def preprocess_desc(desc, is_bert=True):
    desc = _removeNonAscii(desc)
    desc = remove_html(desc)
    if is_bert and len(desc.split())>512:
        desc = remove_stop_words(desc)
    elif not is_bert:
        desc = make_lower_case(desc)
        desc = remove_punctuation(desc)
    return desc

@app.route('/recommend_books', methods=['POST'])
@cross_origin()
def recommend_books_api():
    try:
        json_data = request.json
        title = json_data.get('title', None)
        desc = json_data.get('description', None)
        filter_by_rating = json_data.get('filter_by_rating', True)
        embedding_type = json_data.get('embedding_type', 'bert')
        num_books = json_data.get('num_books', 5)
        if not desc and not title:
            return 'Please provide either title or description', 422
        if desc:
            desc = preprocess_desc(desc, is_bert=embedding_type=='bert')
        # Reading the dat
        data_dict = get_data_dict(embedding_type)
        output = recommend_books(data_dict['embedding_transformer'], data_dict['cosine_similarity_mat'],  title=title, desc=desc, filter_by_rating=filter_by_rating, vector_col=embedding_type+'_vectors', num_books=num_books)
        print(json.dumps(output, indent = 4))
        return jsonify(output)
    except Exception as e:
        print(e)
        return e, 500

if __name__=="__main__":
    app.run(host='0.0.0.0', port=8000)