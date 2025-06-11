import pandas as pd
import numpy as np
import streamlit as st
import io
import re

from spellchecker import SpellChecker
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

# ------------------- Resource & Preprocessing -------------------
@st.cache_resource
def get_preprocessors():
    stemmer = StemmerFactory().create_stemmer()
    stopword_remover = StopWordRemoverFactory().create_stop_word_remover()
    return stemmer, stopword_remover

stemmer, stopword_remover = get_preprocessors()

def preprocess(text):
    text = text.lower()
    text = stopword_remover.remove(text)
    text = stemmer.stem(text)
    return text

spell = SpellChecker()

def correct_typo(text):
    corrected = []
    for word in text.split():
        corrected_word = spell.correction(word)
        corrected.append(corrected_word if corrected_word else word)
    return ' '.join(corrected)

@st.cache_data
def load_and_clean_data():
    df = pd.read_excel("Pariwisata.xlsx")
    df.fillna('', inplace=True)
    df['combined_text'] = (
        df['Place_Name'].astype(str) + " " +
        df['City'].astype(str) + " " +
        df['Category'].astype(str) + " " +
        df['Description'].astype(str)
    )
    df['cleaned_description'] = df['combined_text'].apply(preprocess)
    return df

df = load_and_clean_data()

# ------------------- TF-IDF Variants -------------------
tfidf_options = {
    'Default': TfidfVectorizer(),
    'With Bigrams': TfidfVectorizer(ngram_range=(1, 2)),
    'Sublinear TF': TfidfVectorizer(sublinear_tf=True),
}

@st.cache_data
def get_vectorizer_and_matrix(variant_name, corpus):
    vectorizer = tfidf_options[variant_name]
    matrix = vectorizer.fit_transform(corpus)
    return vectorizer, matrix

# ------------------- Word2Vec -------------------
@st.cache_resource
def train_word2vec(corpus):
    tokenized = [word_tokenize(text) for text in corpus]
    model = Word2Vec(sentences=tokenized, vector_size=100, window=5, min_count=2, workers=4)
    return model

def document_vector(doc, model):
    tokens = [word for word in word_tokenize(doc) if word in model.wv]
    if not tokens:
        return np.zeros(model.vector_size)
    return np.mean(model.wv[tokens], axis=0)

def get_w2v_similarity(query, corpus, model):
    query_vec = document_vector(query, model).reshape(1, -1)
    doc_vecs = np.array([document_vector(doc, model) for doc in corpus])
    return cosine_similarity(query_vec, doc_vecs).flatten()

# ------------------- UI Layout -------------------
st.title("🔍 Sistem Pencarian Informasi Pariwisata")

query = st.text_input("Masukkan kata kunci pencarian:")
variant = st.selectbox("Pilih metode pencarian:", ["TF-IDF", "Word2Vec"])
if variant == "TF-IDF":
    tfidf_choice = st.selectbox("Varian TF-IDF:", list(tfidf_options.keys()))
limit = st.slider("🔢 Jumlah hasil per halaman:", 5, 50, 10, step=5)

categories = sorted(df['Category'].dropna().unique().tolist())
selected_categories = st.multiselect("🏷️ Pilih Kategori:", categories, default=categories)
max_price = st.slider("💰 Batas Harga Maksimal (Rp):", 0, int(df['Price'].max()), value=int(df['Price'].max()))

def highlight_keywords(text, keyword):
    for word in keyword.split():
        pattern = re.compile(re.escape(word), re.IGNORECASE)
        text = pattern.sub(f"<mark>{word}</mark>", text)
    return text

# ------------------- Pencarian -------------------
if query:
    filtered_df = df[
        df['Category'].isin(selected_categories) & (df['Price'] <= max_price)
    ].copy()

    if filtered_df.empty:
        st.warning("⚠️ Tidak ada data yang sesuai dengan filter.")
    else:
        processed_query = preprocess(query)
        
        if variant == "TF-IDF":
            vectorizer, tfidf_matrix = get_vectorizer_and_matrix(tfidf_choice, filtered_df['cleaned_description'])
            query_vec = vectorizer.transform([processed_query])
            similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()
        else:  # Word2Vec
            w2v_model = train_word2vec(filtered_df['cleaned_description'])
            similarity = get_w2v_similarity(processed_query, filtered_df['cleaned_description'], w2v_model)

        filtered_df['score'] = similarity
        results = filtered_df[filtered_df['score'] > 0].sort_values(by='score', ascending=False)

        if results.empty:
            st.error("❌ Tidak ditemukan hasil yang relevan.")
        else:
            st.subheader("📍 Hasil Pencarian")
            total_results = results.shape[0]
            total_pages = (total_results - 1) // limit + 1
            current_page = st.number_input("📄 Halaman:", min_value=1, max_value=total_pages, value=1)
            start_idx = (current_page - 1) * limit
            end_idx = start_idx + limit
            paged_results = results.iloc[start_idx:end_idx]

            for _, row in paged_results.iterrows():
                st.markdown(f"### {row['Place_Name']} - *{row['City']}*")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("⭐ Rating", f"{row['Rating']}")
                col2.metric("🎯 Skor", f"{row['score']:.3f}")
                col3.metric("🏷️ Kategori", row['Category'])
                col4.metric("🎟️ Harga", f"Rp{row['Price']:,}")
                desc = highlight_keywords(row['Description'][:350], query)
                st.markdown(f"""
                <div style="background-color:#f9f9f9; padding:10px; border-left:5px solid #1f77b4; border-radius:5px; margin-bottom:15px">
                <p style="font-size: 14px; line-height: 1.6; color:#333">{desc}...</p>
                </div>
                """, unsafe_allow_html=True)

            buffer = io.BytesIO()
            results.to_excel(buffer, index=False)
            st.download_button("⬇️ Unduh Semua Hasil Pencarian (Excel)", buffer.getvalue(), file_name="hasil_pencarian.xlsx")
else:
    st.info("💡 Masukkan kata kunci untuk mencari informasi pariwisata.")
