import numpy as np
import pandas as pd

file_name_loc = '''Lesson_2_Implementing_Gradient_Descent/implementing_gradient_descent/binary.csv'''

admissions = pd.read_csv(file_name_loc)

# Make dummy variables for rank
dummies_rank = pd.get_dummies(admissions['rank'], prefix='rank')
data = pd.concat([admissions, dummies_rank], axis=1)
data = data.drop('rank', axis=1)

# Standarize features
for field in ['gre', 'gpa']:
    mean, std = data[field].mean(), data[field].std()
    data.loc[:, field] = (data[field] - mean) / std

# Split off random 10% of the data for testing
np.random.seed(42)
sample = np.random.choice(data.index, size=int(len(data) * 0.9), replace=False)
data, test_data = data.iloc[sample], data.drop(sample)

# Split into features and targets
features, targets = data.drop('admit', axis=1), data['admit']
features_test, targets_test = test_data.drop('admit', axis=1), test_data['admit']

features = np.float_(features)
targets = np.float_(targets)

features_test = np.float_(features_test)
targets_test = np.float_(targets_test)