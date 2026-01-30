<H3>ENTER YOUR NAME: SYED MOHAMED RAIHAN M</H3>
<H3>ENTER YOUR REGISTER NO: 212224240167</H3>
<H3>EX. NO.1</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:

### Import libraries
```PYTHON
import pandas as pd
import numpy as np
import seaborn as sns   # for outlier detection
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
```

### Read the dataset directly
```PYTHON
df = pd.read_csv('Churn_Modelling.csv')
print("First 5 rows of the dataset:")
df.head()
```

### Find missing values
```PYTHON
print(df.isnull().sum())
```

### Identify categorical columns
```PYTHON
categorical_cols = df.select_dtypes(include=['object']).columns
print("\nCategorical columns:", categorical_cols.tolist())
```

### Apply Label Encoding to categorical columns
```PYTHON
label_encoder = LabelEncoder()
for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])

print("\nData after encoding:")
print(df.head(5))
```
### Handling missing values only for numeric columns
```PYTHON
for col in df.select_dtypes(include=['float64', 'int64']).columns:
    df[col].fillna(df[col].mean().round(1), inplace=True)

df.isnull().sum()
```

### Detect Outliers (example using seaborn)
```PYTHON
print("\nDetecting outliers (example: CreditScore column):")
sns.boxplot(x=df['CreditScore'])
```

### Example statistics for 'CreditScore'
```PYTHON
print("\nStatistics for 'CreditScore':")
df['CreditScore'].describe()
```

### Splitting features (X) and labels (y)
```PYTHON
X = df.drop('Exited', axis=1).values  # Features (drop target column)
y = df['Exited'].values   

print("\nFeature Matrix (X):")
print(X)
print("\nLabel Vector (y):")
print(y)
```
### Normalizing the features
```PYTHON
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)
```

### First 5 rows after normalization
```PYTHON
pd.DataFrame(X_normalized, columns=df.columns[:-1]).head()
```

### Splitting into Training and Testing Sets
```PYTHON
X_train, X_test, y_train, y_test = train_test_split(
    X_normalized, y, test_size=0.2, random_state=42
)

print("\nShapes of Training and Testing sets:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)
```

## OUTPUT:
<img width="1314" height="299" alt="image" src="https://github.com/user-attachments/assets/da779765-f163-409e-8b89-90bc62bf764d" />

<img width="272" height="353" alt="image" src="https://github.com/user-attachments/assets/6f50ed2b-1bf9-46c4-9c17-1e1b8c0b42d3" />

<img width="616" height="30" alt="image" src="https://github.com/user-attachments/assets/a94f0c7a-6ed4-484a-9d12-c4c09fa9d959" />

<img width="787" height="498" alt="image" src="https://github.com/user-attachments/assets/bab11f8b-d66f-4554-8461-cc5fd5c847c0" />

<img width="342" height="596" alt="image" src="https://github.com/user-attachments/assets/3f4c8b15-ebef-478c-a777-190fe25eadd5" />

<img width="747" height="580" alt="image" src="https://github.com/user-attachments/assets/3eb2bf96-5183-479d-a4a8-360df6a66202" />

<img width="410" height="463" alt="image" src="https://github.com/user-attachments/assets/71db54ed-4748-4cfe-9399-8bff5b26b414" />

<img width="758" height="444" alt="image" src="https://github.com/user-attachments/assets/b01e1cd0-3a4c-4307-821e-0e5b620c503a" />

<img width="1343" height="310" alt="image" src="https://github.com/user-attachments/assets/a7bc5719-3826-46fa-b4bb-3db26ce24657" />

<img width="830" height="327" alt="image" src="https://github.com/user-attachments/assets/e15d1aa6-6202-4c43-8308-ef951a9ea6e4" />


## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


