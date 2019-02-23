
# # Topic Modelling Using Latent Dirichlet Allocation
# ## What does LDA do ?
# It represents documents as mixture of topics that spits out words with certain probabilities.
# It assumes that documents are produced in the following fashion:
# - Choose a topic mixture for the document over K fixed topics.
# - First picking a topic according to the multinomial distribution that we sampled previously
# 
# ### Generate each word in the topic 
# - Using the topic to generate the word itself.
# - Example if the topic is 'food' the probability of generating the word 'apple' is more than 'home'.
# 
# #### LDA then tries to backtrack from these words to topics that might have created these collection of words 
# 
# 
# ## Assumptions for LDA(Latent Dirichlet Allocation) in Topic Modelling
# - Documents are probability distribution over latent topics.
# - Topics themselves are probability distribution over latent words.
# 
# ## Steps to how LDA executed
# - We iterate through every document and assign each word in it to a particular K topic that we defined before.
# - This random assignment gives us topic representation and word distribution of every topic.
# - We iterate over every word in every topic and calculate t 
#     p( t topic|d document) - the proportion of words assigned to each topic in every document d .
# - We iterate over every word in every topic and calculate t :
#     p(w word|d document) - the proportion of assignments to each topic in every document that comes from the word w.  
# - Reassign w to new topic with probability p(t topic|d document) * p(w word|t topic).
# - This is basically that topic t  generated the word w.
# - At the end we have the words with highest probability of being assigned topic. 
# 
# ## Imorting basic libraries

import pandas as pd
npr_csv = pd.read_csv('npr.csv')


npr_csv.head()


# ## Our Articles consist of different types of articles

from sklearn.feature_extraction.text import CountVectorizer


# max_df gets rid of terms that are common in lot of documents(90%)<br>
# min_df minimum document frequency of word to be counted in atleast 2 documents<br>
# to remove stop words 'stop_words = "english"'

cv = CountVectorizer(max_df = 0.9, min_df = 2, stop_words="english")


# To calculate the document term frequency

dtm = cv.fit_transform(npr_csv['Article'])


# Importing our LDA Library

from sklearn.decomposition import LatentDirichletAllocation



LDA = LatentDirichletAllocation(n_components= 7, random_state=42)



LDA.fit(dtm)


# ### These are all the words in our LDA


len(cv.get_feature_names())


# ### Extracting the top 15 words from each topic 


import numpy as np



for index,topic in enumerate(LDA.components_):
    print(f'The top 15 words for topic #{index}')
    print([cv.get_feature_names()[index] for index in topic.argsort()[-15:]])
    print('\n')
    print('\n')



topic_results = LDA.transform(dtm)



npr_csv['Topic'] = topic_results.argmax(axis= 1)



npr_csv

