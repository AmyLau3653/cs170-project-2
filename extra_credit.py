import pandas as pd
from ucimlrepo import fetch_ucirepo 

# fetch dataset 
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17) 
  
# data (as pandas dataframes) 
X = breast_cancer_wisconsin_diagnostic.data.features 
y = breast_cancer_wisconsin_diagnostic.data.targets 
  
# variable information 
print(breast_cancer_wisconsin_diagnostic.variables) 

# rename target column for clarity
y = y.rename(columns={'Diagnosis': 'target'})

# binary encoding
y['target'] = y['target'].map({'M': 1, 'B': 0})

# z-normalize all feature columns
X = (X - X.mean()) / X.std()

# combine features and target into a dataFrame
df = pd.concat([y, X], axis=1)

# show the first few rows
print(df.head())

# target column check
print(df['target'].value_counts())  

# save as a space-separated txt file, without headers (column name) or index
df.to_csv("breast_cancer_data.txt", sep=" ", index=False, header=False)