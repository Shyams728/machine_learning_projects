## Toxic Tweets classification

```python
# Import necessary libraries
import string
import nltk
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix,
    roc_curve, roc_auc_score, precision_recall_curve, average_precision_score, auc
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from imblearn.over_sampling import RandomOverSampler
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from tqdm import tqdm
import time

import matplotlib.pyplot as plt
import seaborn as sns
```

This section imports various libraries and modules needed for data manipulation, natural language processing (NLP), machine learning, and visualization.

```python
# Download NLTK resources if not already downloaded
nltk.download('wordnet')
nltk.download('stopwords')
```

Downloads additional resources for the Natural Language Toolkit (NLTK) library if they haven't been downloaded before.

```python
# Step 1: Load the dataset
df = pd.read_csv('/content/tweet_dataset/FinalBalancedDataset.csv')
```

Loads a CSV dataset into a Pandas DataFrame (`df`).

```python
# Step 2: Data Exploration
# Check class distribution
class_distribution = df['Toxicity'].value_counts()
print("Class Distribution: ", class_distribution)
```

Examines the distribution of classes in the dataset and prints the result.

```python
# Step 3: Data Splitting
# Use cross-validation for model evaluation
from sklearn.model_selection import cross_val_score
```

Imports a cross-validation function from scikit-learn.

```python
# Step 4: Text Preprocessing
# Define functions for removing unwanted characters and preprocessing text
# ...

# Apply preprocessing to the 'tweet' column and create a new 'clean_tweet' column
df['clean_tweet'] = df['tweet'].apply(remove_unwanted_chars).apply(preprocess_text)
```

Defines functions for removing unwanted characters and preprocessing text, then applies these functions to create a new column (`clean_tweet`) in the DataFrame.

```python
# Step 5: Feature Extraction
# Create a TF-IDF representation of text data into numerical vectors
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(df['clean_tweet'])

# Create a Bag of Words (BoW) representation text data into numerical vectors using a CountVectorizer
count_vectorizer = CountVectorizer()
X_bow = count_vectorizer.fit_transform(df['clean_tweet'])
```

Uses TF-IDF and Bag of Words (BoW) vectorization techniques to convert text data into numerical vectors (`X_tfidf` and `X_bow`).

```python
# Define functions for plotting ROC curve and confusion matrix
# ...

def model_training_and_evaluation(x, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Handle imbalanced data using oversampling
    oversampler = RandomOverSampler()
    X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)

    # Define classifiers
    classifiers = [
        DecisionTreeClassifier(random_state=42, criterion='entropy'),
        RandomForestClassifier(random_state=42),
        MultinomialNB(),
        KNeighborsClassifier(),
        SVC(probability=True)
    ]

    results = []
    # Iterate through classifiers
    for classifier in tqdm(classifiers, desc="Processing algorithm", unit="model", total=len(classifiers)):
        model_name = classifier.__class__.__name__

        # Use a pipeline for better organization
        if 'SVC' in model_name:
            model = make_pipeline(StandardScaler(with_mean=False), classifier)
        else:
            model = classifier

        # Fit the classifier on the resampled training data
        model.fit(X_train_resampled, y_train_resampled)

        # Predict on the test data
        y_pred = model.predict(X_test)

        # Calculate various evaluation metrics
        # ...

        # Print and plot metrics
        # ...

        results.append([model_name, precision, recall, f1, confusion_matrix_result, roc_auc])
        time.sleep(0.1)

    # Create a DataFrame to store the results
    results_df = pd.DataFrame(results, columns=['Model', 'Precision', 'Recall', 'F1-Score', 'Confusion Matrix', 'ROC-AUC'])

    return results_df
```

Defines a function for training and evaluating multiple classifiers, handling imbalanced data, and storing and printing the evaluation results. The classifiers include Decision Tree, Random Forest, Naive Bayes, K-Nearest Neighbors, and Support Vector Classifier (SVC).

This function returns a Pandas DataFrame with the evaluation results for each classifier.
