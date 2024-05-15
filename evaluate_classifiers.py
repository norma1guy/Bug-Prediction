import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import cross_val_score, RepeatedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import wilcoxon
from sklearn.metrics import precision_recall_fscore_support
import warnings

# Filter out the specific warning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Your existing code for training classifiers and evaluating them


# Step 1: Load the Feature Vectors
feature_vectors_df = pd.read_csv('modified_feature_vector_list.csv')

# Step 2: Prepare Data and Split into Train and Validation Sets
X = feature_vectors_df.drop(columns=['buggy', 'Class'])
y = feature_vectors_df['buggy']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Define Classifiers
classifiers = {
    "Decision Tree": DecisionTreeClassifier(max_depth=5),
    "Naive Bayes": GaussianNB(),
    "SVM (Linear Kernel)": SVC(kernel='linear', C=1.0),
    "Multi-Layer Perceptron": MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=200),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10)
}

for classifier_name, classifier in classifiers.items():
    print("Training", classifier_name)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_val)
    
    precision, recall, _, _ = precision_recall_fscore_support(y_val, y_pred, average='binary')
    print(classifier_name, "- Precision:", precision)
    print(classifier_name, "- Recall:", recall)
    print("="*40)

results = {}
for classifier_name, classifier in classifiers.items():
    print("Evaluating", classifier_name)
    # Define pipeline with scaler (needed for MLPClassifier)
    pipeline = make_pipeline(StandardScaler(), classifier)
    # Perform cross-validation
    cv_scores_f1 = cross_val_score(pipeline, X, y, cv=RepeatedKFold(n_splits=5, n_repeats=20), scoring='f1')
    cv_scores_precision = cross_val_score(pipeline, X, y, cv=RepeatedKFold(n_splits=5, n_repeats=20), scoring='precision')
    cv_scores_recall = cross_val_score(pipeline, X, y, cv=RepeatedKFold(n_splits=5, n_repeats=20), scoring='recall')
    
    # Store evaluation metrics
    results[classifier_name] = {
        'F-measure': cv_scores_f1,
        'Precision': cv_scores_precision,
        'Recall': cv_scores_recall
    }

    # Print mean and standard deviation of evaluation metrics
    print("Mean F-measure:", np.mean(cv_scores_f1))
    print("Mean Precision:", np.mean(cv_scores_precision))
    print("Mean Recall:", np.mean(cv_scores_recall))
    print("="*40)

# Step 5: Use Wilcoxon test to compare classifiers
wilcoxon_results = pd.DataFrame(index=classifiers.keys(), columns=classifiers.keys())
for classifier_name1, classifier_name2 in itertools.combinations(classifiers.keys(), 2):
    print(f"Comparing {classifier_name1} and {classifier_name2}...")
    _, p_f1 = wilcoxon(results[classifier_name1]['F-measure'], results[classifier_name2]['F-measure'])
    _, p_precision = wilcoxon(results[classifier_name1]['Precision'], results[classifier_name2]['Precision'])
    _, p_recall = wilcoxon(results[classifier_name1]['Recall'], results[classifier_name2]['Recall'])
    
    wilcoxon_results.loc[classifier_name1, classifier_name2] = p_f1
    wilcoxon_results.loc[classifier_name2, classifier_name1] = p_precision
    
    # Check if the difference is statistically significant
    if p_f1 < 0.05:
        print(f"  F-measure difference between {classifier_name1} and {classifier_name2} is statistically significant.")
    if p_precision < 0.05:
        print(f"  Precision difference between {classifier_name1} and {classifier_name2} is statistically significant.")
    if p_recall < 0.05:
        print(f"  Recall difference between {classifier_name1} and {classifier_name2} is statistically significant.")

# Step 6: Display box plots of evaluation metrics
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, metric in enumerate(['F-measure', 'Precision', 'Recall']):
    data = {classifier_name: results[classifier_name][metric] for classifier_name in classifiers.keys()}
    df = pd.DataFrame(data)
    df.boxplot(ax=axes[i])
    axes[i].set_title(f'{metric} Box Plot')

plt.tight_layout()
plt.show()

# Step 7: Display p-values of pairwise Wilcoxon test
print("P-values of pairwise Wilcoxon test:")
print(wilcoxon_results)