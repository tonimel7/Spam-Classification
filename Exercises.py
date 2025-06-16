#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import fetch_openml
import pandas as pd
import numpy as np 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split


# **Exercise 1**

# In[2]:


mnist = fetch_openml("mnist_784", as_frame=False)
X, y = mnist.data, mnist.target


# In[3]:


X.shape, y.shape


# In[4]:


X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]


# In[5]:


knn_clf = KNeighborsClassifier()


# In[6]:


params = {"weights": ["uniform", "distance"],
         "n_neighbors": [3,5,10]}

knn_cv = RandomizedSearchCV(knn_clf, params, cv = 3, scoring = "accuracy")
knn_cv.fit(X_train, y_train)


# **Exercise 4**

# In[7]:


from pathlib import Path
import email
from bs4 import BeautifulSoup


# In[8]:


main_dir = Path("C:\\Users\\melis\\Desktop\\BLF\\Chapter 3 - Classification\\corpus")


# In[9]:


def extract_text_from_email(raw_email: str) -> str:

    #This turns the long string into a structured email object
    msg = email.message_from_string(raw_email)


    parts = [] # empty list to append the email parts 

    for part in msg.walk():# for every PART in a Message object email (HTML, plain text, etc.)
        ctype = part.get_content_type() # see what type of content the Message object part is 
        payload = part.get_payload(decode=True) # Get the Message object part and decode it into raw bytes to remove any transport encodings

        if payload is None: # If there is no payload (Message object is empty)
            continue # continue and leave the loop  

        text = payload.decode("latin-1", errors="ignore") #decode the text in latin characters and ignore errors from 
        # weird characters

        if ctype == "text/html": # if the content type of the Message object part is HTML, 
            text = BeautifulSoup(text, "html.parser") # parse it with BeautifulSoup, which enables us to:
            text = text.get_text() # get the actual meaningful text from that part. 


        if ctype in ("text/plain", "text/html"): # if the content type is either plain text or html text
            parts.append(text) # append the text to the parts list 

    return "\n".join(parts).strip() # Return the parts, joined together, and remove any whitespace. 1 string per entire email 

# *: Emails parts come wrapped in MIME transfer encodings (standard that lets email carry more than plain text).
# We decode these to get raw bytes and then maps each raw byte to the corresponding Latin-1 chart.


# In[10]:


records = []

# 1) GO TO EACH FOLDER, 
# 2) ITERATE OVER EACH FILE, AND FOR EVERY FILE, READ THE TEXT AND APPLY THE FUNCTION TO EXTRACT THE MEANINGFUL PART OF THE TEXT 
# 3) APPEND THE EXTRACTED PART IN DICTIONARY ALONG WITH THE LABEL (FOR EACH FILE)

for label in ("spam", "ham"): # going over the 2 sub folders

    folder = main_dir / label # the 2 folders are: path/spam or path/ham 

    for file_path in folder.iterdir(): #going to the actual subfolder directory 

        if not file_path.is_file(): # if the file path is not a file, go to the next loop. (???)
            continue

        raw = file_path.read_text(encoding="latin-1", errors="ignore") # read the text from the file.

        body = extract_text_from_email(raw) # take the text to the function to extract meaningful part of the email.

        records.append({"text": body,
                       "label": label}) # create a dictionary with text and label for classification


# In[14]:


df = pd.DataFrame(records)
df


# In[15]:


df_num = df


# In[16]:


for i in range(len(df)):
    df_num.loc[i,"label"] = 1 if df.loc[i,"label"] == "spam" else 0 


# In[17]:


df_num["label"].value_counts() # slightly imbalanced, 501 spams and 2501 normal messages.


# We have now transformed the multiple email files in the spam & ham subfolders into a dataframe that contains 
# 
# ALL emails and their LABELS

# In[18]:


from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer

stop_words = set(stopwords.words("english"))


# In[19]:


df_num


# In[74]:


# remove_stopwords function is designed to work email by email. 

def remove_stopwords(text):
    cleaned_words = [] # the clean words will be added here for every row 
    stop_words = set(stopwords.words('english')) 

    for word in word_tokenize(text, language = "english"): # for every word in text, tokenize it in words based on english language 
        word = word.lower() # lower the word 

        if word.isalpha() and word not in stop_words: # if word is alphabetic AND not a stopword,
            cleaned_words.append(word) # append the word back to the list 

    return cleaned_words # return the cleaned list of words



def clean_series(X):
    # X(i) is raw text email. The first line applies the remove_stopwords function to i 

    # second line appends the tokens as strings with an empty space in between

    token_list = X.apply(remove_stopwords) # token_list is a Series of lists
    cleaned_string = token_list.str.join(" ") # this pandas takes EACH list and concatenates its items into 
    # one string, placing a space in between. We need them in 1 string. 
    return cleaned_string
    #return cleaned_string.to_frame() # this line converts to 2D because we need it in 2D to avoid some weird error later


# In[75]:


# Transformer to apply stopword & number removal, and return a clean string. 
stopword_transformer = FunctionTransformer(clean_series, validate=False)


# In[90]:


#YOOOO
# This pipeline handles stopword and number removal, and transforms into a DTM. 

clean_and_vect = make_pipeline(
    stopword_transformer,
    CountVectorizer()
)


# In[91]:


# This transformation applies the pipeline in the text column 
ct = ColumnTransformer(
    transformers = [
        ("clean_email", clean_and_vect, "text")], 
    remainder="drop")

#ct.set_output(transform="pandas")


# In[99]:


from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression

log_reg_pipeline = make_pipeline(ct, 
                        TruncatedSVD(n_components=150, random_state=42),
                        LogisticRegression())


# In[125]:


X = df_num[["text"]]

y = df_num["label"]
y = y.astype(int)


# In[126]:


X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.3, stratify = y)


# In[127]:


y_train.values.ravel()


# In[128]:


log_reg_pipeline.fit(X_train, y_train)


# In[130]:


y_pred = log_reg_pipeline.predict(X_test)


# In[133]:


from sklearn.metrics import accuracy_score 

accuracy_score(y_pred, y_test)


# In[ ]:




