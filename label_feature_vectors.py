import os
import pandas as pd

# Step 1: Read the feature_vector_list.csv and extract class names
feature_vector_df = pd.read_csv('feature_vector_list.csv')

# Step 2: Read the .src files and extract class names
src_files_dir = 'resources/modified_classes/'
class_names_from_src = []
for root, dirs, files in os.walk(src_files_dir):
    for file in files:
        if file.endswith('.src'):
            with open(os.path.join(root, file), 'r') as f:
                class_names_from_src.extend([class_name.split('.')[-1] for class_name in f.read().splitlines()])

# Step 3: Compare class names and assign buggy column
buggy_classes = []
for class_name in feature_vector_df['Class']:
    if class_name in class_names_from_src:
        buggy_classes.append(1)
    else:
        buggy_classes.append(0)

# Add the 'buggy' column to the feature_vector_df
feature_vector_df['buggy'] = buggy_classes

# Step 4: Write the modified data to a new CSV
feature_vector_df.to_csv('modified_feature_vector_list.csv', index=False)
