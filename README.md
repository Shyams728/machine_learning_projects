## Kannada digits 

1. **Import Libraries:**
   - `numpy` and `pandas` for data manipulation.
   - `train_test_split` for splitting the dataset into training and testing sets.
   - Various metrics and classifiers from `sklearn` for model evaluation and training.
   - `PCA` for dimensionality reduction.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
```

2. **Load Data:**
   - `load_data` function is defined to load data from NPZ files.
   - Training and testing data along with labels are loaded using this function.

```python
# Function to load data from NPZ files
def load_data(file_path):
    data = np.load(file_path)
    return data['arr_0']

# Load training and testing data
train_data = load_data('/content/drive/MyDrive/data/Kannada_MNIST_datataset_paper/Kannada_MNIST_npz/Kannada_MNIST/X_kannada_MNIST_train.npz')
test_data = load_data('/content/drive/MyDrive/data/Kannada_MNIST_datataset_paper/Kannada_MNIST_npz/Kannada_MNIST/X_kannada_MNIST_test.npz')

# Load labels
y_train = load_data('/content/drive/MyDrive/data/Kannada_MNIST_datataset_paper/Kannada_MNIST_npz/Kannada_MNIST/y_kannada_MNIST_train.npz')
y_test = load_data('/content/drive/MyDrive/data/Kannada_MNIST_datataset_paper/Kannada_MNIST_npz/Kannada_MNIST/y_kannada_MNIST_test.npz')
```

3. **Data Preprocessing:**
   - `flatten_data` function is defined to flatten the image data.

```python
# Flatten the image data
def flatten_data(data):
    return data.reshape(data.shape[0], -1)
```

4. **PCA Dimensionality Reduction:**
   - `apply_pca` function is defined to apply PCA with the desired number of components.

```python
# Initialize PCA with the desired number of components
def apply_pca(data, n_components):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(data)
```

5. **Classifier Training and Evaluation:**
   - `train_and_evaluate_classifier` function is defined to train a classifier and return various evaluation metrics.
   - ROC-AUC scores and data for ROC curves are calculated.
   - Confusion matrix, precision, recall, F1-score, and weighted ROC-AUC are calculated.

```python
# Train a classifier and return evaluation metrics
def train_and_evaluate_classifier(classifier, X_train, y_train, X_test, y_test):

    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    # Calculate ROC-AUC score
    y_prob = classifier.predict_proba(X_test)

    # Define the class names based on your dataset
    class_names = np.unique(y_train)
    # Check the shape of y_prob
    if y_prob.ndim == 1:
        # If it's a 1D array, reshape it to a 2D array
        y_prob = y_prob.reshape(-1, 1)

    # Calculate ROC-AUC and prepare data for ROC curve plotting
    n_classes = len(np.unique(y_test))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve((y_test == i).astype(int), y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Prepare ROC curve data
    roc_auc_data = {
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': roc_auc,
        'class_names': class_names,
        'n_classes': n_classes
    }

    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    confusion_matrix_result = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted')

    return precision, recall, f1, confusion_matrix_result, roc_auc, roc_auc_data
```

6. **Classifier Experimentation:**
   - The code then splits the data into training and testing sets.
   - A list of classifiers is defined.
   - A loop is used to experiment with different PCA component sizes and different classifiers.

```python
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(train_data, y_train, test_size=0.2, random_state=42)

# Define a list of classifiers to experiment with
classifiers = [
    DecisionTreeClassifier(random_state=42, criterion='entropy'),
    RandomForestClassifier(random_state=42),
    MultinomialNB(),
    KNeighborsClassifier(),
    SVC(probability=True)
]
```

7. **Results Storage and Printing:**
   - The code prints and stores results, including precision, recall, F1-score, ROC-AUC, ROC curve data, and confusion matrix.

```python
# Dictionary to store results
results_dict = {}

# Experiment with different PCA component sizes
component_sizes = [10, 15, 20, 25]

for n_components in component_sizes:
    X_train_pca = apply_pca(flatten_data(X_train), n_components)
    X_test_pca = apply_pca(flatten_data(X_test), n_components)

    results = []
    for classifier in classifiers:
        model_name = classifier.__class__.__name__

        if model_name == 'MultinomialNB':
            # Use the MultinomialNB classifier without PCA for this specific case
            precision, recall, f1, confusion_matrix_result, roc_auc, roc_auc_data = train_and_evaluate_classifier(classifier, flatten_data(X_train), y_train, flatten_data(X_test), y_test)
        else:
            # For other classifiers, use PCA as before
            precision, recall, f1, confusion_matrix_result, roc_auc, roc_auc_data = train_and_evaluate_classifier(classifier, X_train_pca, y_train, X_test_pca, y_test)

        print(f"{model_name} (PCA-{n_components}) - Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}, ROC AUC: {roc_auc:.4f}")
        results.append([model

_name, precision, recall, f1, roc_auc, roc_auc_data, confusion_matrix_result])

    results_dict[f'PCA-{n_components}'] = pd.DataFrame(results, columns=['Model', 'Precision', 'Recall', 'F1-Score', 'ROC AUC', 'ROC AUC Data', 'Confusion Matrix'])
```


