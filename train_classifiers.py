import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support

# Step 1: Load the Feature Vectors
feature_vectors_df = pd.read_csv('modified_feature_vector_list.csv')

# Step 2: Prepare Data and Split into Train and Validation Sets
X = feature_vectors_df.drop(columns=['buggy'])
y = feature_vectors_df['buggy']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train,X_val,y_train,y_val)

# Step 3: Hyperparameter Tuning and Training
classifiers = {
    "Decision Tree": DecisionTreeClassifier(max_depth=5),
    "Naive Bayes": GaussianNB(),
    "SVM (Linear Kernel)": SVC(kernel='linear', C=1.0),
    "Multi-Layer Perceptron": MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=200),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10)
}

for classifier_name, classifier in classifiers.items():
    print("Training", classifier_name)
    X_train = X_train.drop(columns=['Class'])
    X_val = X_val.drop(columns=['Class'])
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_val)
    

    precision, recall, _, _ = precision_recall_fscore_support(y_val, y_pred, average='binary')
    print(classifier_name, "- Precision:", precision)
    print(classifier_name, "- Recall:", recall)
    print("="*40)
