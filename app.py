# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 10:52:06 2021

@author: rejid4996
"""

import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer;
from sklearn.metrics.pairwise import cosine_similarity
model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
stop_words = "english"
import numpy as np
import itertools

def max_sum_sim(doc_embedding, candidate_embeddings, candidates, top_n, nr_candidates):
    # Calculate distances and extract keywords
    distances = cosine_similarity(doc_embedding, candidate_embeddings)
    distances_candidates = cosine_similarity(candidate_embeddings, 
                                            candidate_embeddings)

    # Get top_n words as candidates based on cosine similarity
    words_idx = list(distances.argsort()[0][-nr_candidates:])
    words_vals = [candidates[index] for index in words_idx]
    distances_candidates = distances_candidates[np.ix_(words_idx, words_idx)]

    # Calculate the combination of words that are the least similar to each other
    min_sim = np.inf
    candidate = None
    for combination in itertools.combinations(range(len(words_idx)), top_n):
        sim = sum([distances_candidates[i][j] for i in combination for j in combination if i != j])
        if sim < min_sim:
            candidate = combination
            min_sim = sim

    return [words_vals[idx] for idx in candidate]

def mmr(doc_embedding, word_embeddings, words, top_n, diversity):

    # Extract similarity within words, and between words and the document
    word_doc_similarity = cosine_similarity(word_embeddings, doc_embedding)
    word_similarity = cosine_similarity(word_embeddings)

    # Initialize candidates and already choose best keyword/keyphras
    keywords_idx = [np.argmax(word_doc_similarity)]
    candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

    for _ in range(top_n - 1):
        # Extract similarities within candidates and
        # between candidates and selected keywords/phrases
        candidate_similarities = word_doc_similarity[candidates_idx, :]
        target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)

        # Calculate MMR
        mmr = (1-diversity) * candidate_similarities - diversity * target_similarities.reshape(-1, 1)
        mmr_idx = candidates_idx[np.argmax(mmr)]

        # Update keywords & candidates
        keywords_idx.append(mmr_idx)
        candidates_idx.remove(mmr_idx)

    return [words[idx] for idx in keywords_idx]

def main():
    
    """NLP App with Streamlit"""
    
    st.sidebar.title("Topic Extraction App")
    st.sidebar.success("Please reach out to https://www.linkedin.com/in/deepak-john-reji/ for more queries")
    st.sidebar.subheader("Topic Extraction using Embeddings")
    
    st.info("For more contents subscribe to my Youtube Channel https://www.youtube.com/channel/UCgOwsx5injeaB_TKGsVD5GQ")
        
    text3 = st.text_area("copy your paragraph here", "")
    
    n_gram_value = st.sidebar.slider('Select a range of n-gram', 1, 4, 2, 1)
    top_n = st.sidebar.slider('Select the no. of results required', 1, 7, 5, 1)

    
    method = st.sidebar.selectbox('select the method to extract', ('Embeddings', 'Embeddings + Max Sum Similarity', 'Embeddings + Maximal Marginal Relevance'))
    
    if(st.button("Extract")):

        n_gram_range = (n_gram_value, n_gram_value)
                
        # Extract candidate words/phrases
        count = CountVectorizer(ngram_range=n_gram_range, stop_words=stop_words).fit([text3])
        candidates = count.get_feature_names()
        
        # embddings
        doc_embedding = model.encode([text3])
        candidate_embeddings = model.encode(candidates)
        
        #top_n = 5
        
      
        if method == 'Embeddings':
            distances = cosine_similarity(doc_embedding, candidate_embeddings)
            keywords = [candidates[index] for index in distances.argsort()[0][-top_n:]]     
            
        elif method == 'Embeddings + Max Sum Similarity':
            keywords=  max_sum_sim(doc_embedding, candidate_embeddings, candidates, top_n, nr_candidates=10)
     
        elif method == 'Embeddings + Maximal Marginal Relevance':
            keywords = mmr(doc_embedding, candidate_embeddings, candidates, top_n, diversity=0.2)
        st.write(keywords)

# calling the main function
if __name__ == "__main__":
    main()
