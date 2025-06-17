from sklearn.datasets import fetch_openml
import pandas as pd
import numpy as np 
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import loguniform, uniform, randint

from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, recall_score

from sklearn.decomposition import TruncatedSVD # we reduce dimensions to improve performance of classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from pathlib import Path
import email
from bs4 import BeautifulSoup

# setting up the folder directory 
main_dir = Path("C:\\Users\\melis\\Desktop\\BLF\\Chapter 3 - Classification\\corpus")


# This is a function to extract the meaningful email text from the raw email. 
# It's designed to work on a string of raw text extracted from a raw email file. 

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
        # any weird characters that appear

        if ctype == "text/html": # if the content type of the Message object part is HTML, 
            text = BeautifulSoup(text, "html.parser") # parse it with BeautifulSoup, which enables us to:
            text = text.get_text() # get the actual meaningful text from that part. 


        if ctype in ("text/plain", "text/html"): # if the content type is either plain text or html text
            parts.append(text) # append the text to the parts list 

    return "\n".join(parts).strip() # Return the parts, joined together, and remove any whitespace. 1 string per entire email 

# *: Emails parts come wrapped in MIME transfer encodings (standard that lets email carry more than plain text).
# We decode these to get raw bytes and then maps each raw byte to the corresponding Latin-1 chart.

#################################################################
# 1) GO TO EACH FOLDER, 
# 2) ITERATE OVER EACH FILE, AND FOR EVERY FILE, READ THE TEXT AND APPLY THE FUNCTION TO EXTRACT THE MEANINGFUL PART OF THE TEXT (above function)
# 3) APPEND THE EXTRACTED PART IN DICTIONARY ALONG WITH THE LABEL (FOR EACH FILE)

records = []

for label in ("spam", "ham"): # iterating over the 2 sub folders

    folder = main_dir / label # the 2 folders are: path/spam or path/ham 

    for file_path in folder.iterdir(): # going to the actual subfolder that contains each email

        if not file_path.is_file(): # if the file path is not a file, go to the next loop. (???)
            continue

        raw = file_path.read_text(encoding="latin-1", errors="ignore") # gather the entire text from the file.

        body = extract_text_from_email(raw) # take the text to the function to extract meaningful part of the email.

        records.append({"text": body,
                       "label": label}) # create a dictionary with text and label for classification
        
############################################################

df = pd.DataFrame(records)

print(">> Data head and tail:")

print(df.head())
print(df.tail())

print(">> Initial data shape: ", df.shape)

df_num = df

for i in range(len(df)):
    df_num.loc[i,"label"] = 1 if df.loc[i,"label"] == "spam" else 0 

print(">> value counts of target class: ")
print(df_num["label"].value_counts()) # slightly imbalanced, 501 spams and 2501 normal messages.

######################################################################################################################################

# We have now transformed the multiple email files in the spam & ham subfolders into a dataframe that contains 
# ALL emails in 1 column and their LABELS (spam=1, normal=0) in another column 

from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer

stop_words = set(stopwords.words("english"))

#### remove_stopwords function is designed to work email by email (email_1_text, email_2_text, ....) 

def remove_stopwords(text):
    cleaned_words = [] # empty list to add cleaned tokenized words 

    for word in word_tokenize(text, language = "english"): # for every word in text, tokenize it in words based on english language 
        word = word.lower() # lower the word 

        if word.isalpha() and word not in stop_words: # if word is alphabetic AND not a stopword,
            cleaned_words.append(word) # append the word to the list 

    return cleaned_words # return the cleaned list of words

def clean_series(X):
    # X(i) is raw text email. The first line applies the remove_stopwords function to i 

    # second line appends the tokens as strings with an empty space in between

    token_list = X.apply(remove_stopwords) # token_list is a Series of lists
    cleaned_string = token_list.str.join(" ") # this pandas takes EACH list and concatenates its items into one string, 
    # placing a space in between. We need the clean text in 1 string (not list of words) for CountVectorizer to accept it. 

    return cleaned_string


### Setting up the X and y parts 
X = df_num[["text"]]

y = df_num["label"]
y = y.astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.3, stratify = y)


# Transformer to apply stopword & number removal, and return a clean string. 
stopword_transformer = FunctionTransformer(clean_series, validate=False)


# The below pipeline handles stopword and number removal, and transforms into a Document Term Matrix (DTM). 

clean_and_vect = make_pipeline(
    stopword_transformer,
    CountVectorizer()
)

# This transformation applies the pipeline in the email text column 
ct = ColumnTransformer(
    transformers = [
        ("clean_email", clean_and_vect, "text")], 
    remainder="drop")

trunc_svd = TruncatedSVD(n_components=200, random_state=42)

preprocessing_pipeline = make_pipeline(ct, 
                        trunc_svd, 
                        StandardScaler()) # we apply standardization because SVD outputs are unscaled - bad for some classifiers

# Pipeline to apply the preprocessing process and apply LogisticRegression

log_reg_pipeline = make_pipeline(preprocessing_pipeline, 
                        LogisticRegression())

preprocessing_pipeline.fit(X_train)

svd_fitted = preprocessing_pipeline.named_steps["truncatedsvd"]

print(f">> Variance captured by Trunc. SVD at 200 components: {svd_fitted.explained_variance_ratio_.sum()}")


### RANDOMIZED SEARCH: FIND BEST PARAMETERS for LOGIT REGRESSION

params_log= {
    "logisticregression__C": loguniform(1e-4, 1e4),
    "logisticregression__class_weight" :["balanced", None]
}

log_reg_search = RandomizedSearchCV(
    estimator=log_reg_pipeline,
    param_distributions=params_log,
    cv=3,
    refit=True,
    scoring="f1",
    random_state=42,
    n_jobs=-1
)

log_reg_search.fit(X_train, y_train)

print(">> Randomized search CV - logistic regression - best params: ", log_reg_search.best_params_)

####
# Flattening because of requirements of some function later on 
y_train.values.ravel()

# Function to cross evaluate model on the train set 
def evaluate_cv(model, name=None, X=None, y=None, folds=3):
    scoring =["precision", "recall"]

    cv = StratifiedKFold(n_splits=folds)

    scores = cross_validate(model, X, y,cv=cv, 
                   scoring=scoring, n_jobs=-1)
    
    mean_precision= scores["test_precision"].mean()
    mean_recall = scores["test_recall"].mean()
    mean_f1 = 2 / ((1/mean_precision) + (1/mean_recall))

    print(f"The CV mean precision and mean recall of the model {name} are {mean_precision}, {mean_recall} respectively")
    print(f"The CV mean F1 score of the model {name} is {mean_f1}")

evaluate_cv(model = log_reg_search, name = "Logistic Regression (with truncated SVD)", X = X_train, y = y_train)

### RANDOM FOREST CLASSIFIER

rf_pipeline = make_pipeline(preprocessing_pipeline, 
                            RandomForestClassifier())


rf_params = {"randomforestclassifier__min_samples_split": [2, 4, 6],
             "randomforestclassifier__n_estimators": [100,130,180],
             "randomforestclassifier__max_depth": [8, 12, 16]}


rf_search = RandomizedSearchCV(rf_pipeline, 
                               rf_params, 
                               cv = 3,
                               refit=True,
                               scoring="f1",
                               random_state=42,
                               n_jobs=-1)

rf_search.fit(X_train, y_train)

print(">> Randomized search CV - Random Forest clf. - best params: ", rf_search.best_params_)

evaluate_cv(model = rf_search, name = "Random Forest Classifier", X=X_train, y=y_train)


