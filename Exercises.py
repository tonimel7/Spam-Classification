from sklearn.datasets import fetch_openml
import pandas as pd
import numpy as np 
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import loguniform, uniform, randint
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.model_selection import cross_validate, StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, recall_score, precision_recall_curve
from sklearn.decomposition import TruncatedSVD # we reduce dimensions to improve performance of classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from pathlib import Path
import email
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer

""""
The following code turns folders of raw "spam" and "ham"(normal) emails into a labeled DataFrame. After some preprocessing,
the aim is to apply Binary Classification techniques to predict if an email is spam or not based on text attributes. Classifiers applied are
Logistic Regression, Random Forest Classifier and Gaussian Naive bayes. Logistic Regression is the best individual model with over 90% precision 
and recall. 
"""

# Folder directory containing the email files. Try/except is put here in case the code is run in an interactive environment
# where the first base_dir= does not work.
try:    
    base_dir = Path(__file__).resolve().parent
except NameError:
    base_dir = Path().cwd() # if running in interactive env. , just return the working directory

corpus_dir = base_dir / "corpus" # corpus_dir contains the 2 sub folders: spam / ham

# extract_text_from_email will extract the meaningful email text from the raw email. 
# It's designed to work on a string of raw text extracted from a raw email file. 

def extract_text_from_email(raw_email: str) -> str:

    #This turns the long string into a structured email object
    msg = email.message_from_string(raw_email)

    parts = [] # empty list to append the email parts 

    for part in msg.walk():# for every PART in a Message object email (HTML, plain text, etc.)

        ctype = part.get_content_type() # see what type of content the Message object part is 
        payload = part.get_payload(decode=True) # Get the Message object part and decode it into raw bytes to remove any transport encodings *

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

# *: Emails parts come wrapped in MIME transfer encodings (things that lets email carry more than plain text).
# We decode these to get raw bytes and then map each raw byte to the corresponding Latin-1 chart.


## Function to apply Stratified Cross Validation on the train set, will be used for each model
def evaluate_cv(model, name=None, X=None, y=None, folds=3):
    scoring =["precision", "recall"]

    cv = StratifiedKFold(n_splits=folds)

    scores = cross_validate(model, X, y,cv=cv, 
                   scoring=scoring, n_jobs=2)
    
    mean_precision= scores["test_precision"].mean()
    mean_recall = scores["test_recall"].mean()
    mean_f1 = 2 / ((1/mean_precision) + (1/mean_recall))

    print(f"The CV mean precision and mean recall of the model {name} are {mean_precision}, {mean_recall} respectively")
    print(f"The CV mean F1 score of the model {name} is {mean_f1}")


##Function to plot the Out Of Fold (oof) Precision-Recall curve.
def PR_curve_oof(model, X, y, name=None):
    cv = StratifiedKFold(n_splits=3)

    scores_oof = cross_val_predict( # out of fold prediction scores for X.
        model, X, y,
        cv=cv,
        method = "predict_proba", n_jobs=2
    )[:,1] 

    precisions, recalls, thresholds = precision_recall_curve(y_train, scores_oof)

    #Plotting the curve
    plt.plot(recalls, precisions, label=name)
    plt.xlabel("Recall")
    plt.ylabel("Precisions")
    plt.title(f"Precision Recall (PR) Curve - {name}")
    plt.grid(True)
    plt.legend(loc="lower left")
    plt.show(block=False)

#################################################################
# 1) GO TO EACH FOLDER, 
# 2) ITERATE OVER EACH FILE, AND FOR EVERY FILE, READ THE TEXT AND APPLY THE FUNCTION TO EXTRACT THE MEANINGFUL PART OF THE TEXT (above function)
# 3) APPEND THE EXTRACTED PART IN DICTIONARY ALONG WITH THE LABEL (FOR EACH FILE)

records = []

for label in ("spam", "ham"): # iterating over the 2 sub folders

    folder = corpus_dir / label # the 2 folders are: path/spam or path/ham 

    for file_path in folder.iterdir(): # going to the actual subfolder that contains each email and iterating over the email files

        raw = file_path.read_text(encoding="latin-1", errors="ignore") # gather the entire text from the file.

        body = extract_text_from_email(raw) # take the text to the function to extract meaningful part of the email.

        records.append({"text": body,
                       "label": label}) # create a dictionary with text and label for classification
        
############################################################

## Printing some basic informatory things
df = pd.DataFrame(records)

print(">> Data head and tail:")

print(df.head())
print(df.tail())

print(">> Initial data shape: ", df.shape)

## Changing label from SPAM/ HAM to 1 and 0 RESPECTIVELY
df_num = df
for i in range(len(df)):
    df_num.loc[i,"label"] = 1 if df.loc[i,"label"] == "spam" else 0 

print(">> value counts of target class: ")
print(df_num["label"].value_counts()) # slightly imbalanced, 501 spams and 2501 normal messages. We will use stratification

######################################################################################################################################

## We have now transformed the multiple email files in the spam & ham subfolders into a dataframe that contains 
## ALL email texts in 1 column and their LABELS (spam=1, normal=0) in another column 

## Setting up the X and y parts of the data
X = df_num[["text"]]

y = df_num["label"]
y = y.astype(int)

## TEMPORARY MEASURE: Keeping 70% of the original data for faster compute 
# shuffling first because data is highly ordered 
X = shuffle(X, random_state=42)
y = shuffle(y, random_state=42)

X = X[:int(0.7* len(X))]
y = y[:int(0.7*len(y))]

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.3, stratify = y, random_state=42)

vect = TfidfVectorizer(stop_words="english") # stop_words="english" to remove stopwords.

## tfidf vectorizer removes stopwords and transforms into a Document Term Matrix (DTM) by itself. 
clean_and_vect = make_pipeline(vect)

## This transformation applies the pipeline in the email text column 
ct = ColumnTransformer(
    transformers = [
        ("clean_email", clean_and_vect, "text")],  
    remainder="drop")

## Truncated SVD: reduce sparse data into less dimensions
trunc_svd = TruncatedSVD(n_components=500, random_state=42)

preprocessing_pipeline = make_pipeline(ct,
                        trunc_svd)


## Pipeline to apply preprocessing and then -> LogisticRegression
log_reg_pipeline = make_pipeline(preprocessing_pipeline, 
                        LogisticRegression(random_state=42))

## LOGISTIC REGRESSION: RANDOMIZED SEARCH CV
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
    n_jobs=2
)

log_reg_search.fit(X_train, y_train)

print(">> Randomized search CV - logistic regression - best params: ", log_reg_search.best_params_)

svd_only = (log_reg_search.best_estimator_
            .named_steps["pipeline"]
            .named_steps["truncatedsvd"])

print(f"The explained variance captured by Trunc. SVD is: {svd_only.explained_variance_ratio_.sum()}")

y_train.values.ravel() # Flattening because of requirements of some function later on 

## CV scores for Logistic Regression
evaluate_cv(model = log_reg_search, name = "Logistic Regression (with truncated SVD)", X = X_train, y = y_train)

cv = StratifiedKFold(n_splits=3)

PR_curve_oof(log_reg_search, X_train, y_train, name="Logistic Regression")


## RANDOM FOREST CLASSIFIER: RANDOMZIED SEARCH CV
rf_pipeline = make_pipeline(preprocessing_pipeline, 
                            RandomForestClassifier(random_state=42))

rf_params = {"randomforestclassifier__min_samples_split": [2, 4, 6],
             "randomforestclassifier__n_estimators": [120,150,200],
             "randomforestclassifier__max_depth": [8, 12, 16]}

rf_search = RandomizedSearchCV(rf_pipeline, 
                               rf_params, 
                               cv = 3,
                               refit=True,
                               scoring="recall",
                               random_state=42,
                               n_jobs=2)

rf_search.fit(X_train, y_train)

PR_curve_oof(rf_search, X_train, y_train, name="Random Forest Classifier")


print(">> Randomized search CV - Random Forest clf. - best params: ", rf_search.best_params_)

## CV scores for Random Forest Classifier 
evaluate_cv(model = rf_search, name = "Random Forest Classifier", X=X_train, y=y_train)


### GAUSSIAN NAIVE BAYES CLASSIFIER 

naive_bayes_pipeline = make_pipeline(preprocessing_pipeline, 
                                     GaussianNB())

naive_bayes_pipeline.fit(X_train, y_train)

evaluate_cv(model=naive_bayes_pipeline, name ="Gaussian Naive Bayes Classifier", X=X_train, y=y_train)



