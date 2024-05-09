import os
import pandas as pd
import numpy as np

# Step 1: Read the feature_vector_list.csv and extract class names
feature_vector_df = pd.read_csv('feature_vector_list.csv')
classes_in_feature_vector = feature_vector_df['Class'].tolist()

# Step 2: Read the .src files and extract class names
src_files_dir = 'resources/modified_classes/'
class_names_from_src = []
for root, dirs, files in os.walk(src_files_dir):
    for file in files:
        if file.endswith('.src'):
            with open(os.path.join(root, file), 'r') as f:
                class_names_from_src.extend([class_name.split('.')[-1] for class_name in f.read().splitlines()])

# Step 3: Compare class names
if class_names_from_src:
    buggy_classes = [1 if class_name in classes_in_feature_vector else 0 for class_name in class_names_from_src]
else:
    print("No classes found in the .src files.")
    buggy_classes = [0] * len(classes_in_feature_vector)

# Convert to integers explicitly
buggy_classes = [int(value) for value in buggy_classes]

# Handle NaN values
#buggy_classes = [0 if np.isnan(value) else value for value in buggy_classes]

# Step 4: Add the 'buggy' column to the feature_vector_df
feature_vector_df['buggy'] = pd.Series(buggy_classes)

# Step 5: Write the modified data to a new CSV
feature_vector_df.to_csv('modified_feature_vector_list.csv', index=False, float_format='%.0f')

