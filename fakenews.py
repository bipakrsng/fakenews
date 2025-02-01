# %% 
#importing the dependencies
import numpy as np
import pandas as pd
import re  # for searching text in news
from nltk.corpus import stopwords  # from natural language toolkit importing stopwords
from nltk.stem.porter import PorterStemmer  # it takes the word and returns the root word
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk

# %%
# Downloading the required NLTK data
nltk.download('stopwords')
stopwords.words('english')
# print(stopwords.words('english'))

# %%
# Loading the dataset into a pandas DataFrame
news_dataset = pd.read_csv('train.csv')
print(news_dataset.shape)

# %%
# Displaying the first 5 rows of the dataset
pd.set_option('display.max_columns', None)
# print(news_dataset.head())

# %%
# Counting the missing values in the dataset
# print(news_dataset.isnull().sum())

# %%
# Replacing missing values with an empty string
news_dataset = news_dataset.fillna('')
# print(news_dataset)

# %%
# Merging the author name and news title
news_dataset['content'] = news_dataset['author'] + ' ' + news_dataset['title']
news_dataset['content'] = news_dataset['content'].astype(str)

# print(news_dataset['content'])

# %%
# Separating the data and labels
X = news_dataset.drop(columns='label', axis=1)
Y = news_dataset['label']

# print(X)
# print(Y)

# %%
# Stemming: Reducing a word to its root word
# Example: actor, actress, acting --> act

port_stem = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

# %%
# Applying stemming to the 'content' column
news_dataset['content'] = news_dataset['content'].apply(stemming)

# %%
print(news_dataset['content'])


# %%
# Separating the data and labels
X = news_dataset['content'].values
Y = news_dataset['label'].values
print(X)
print(Y)

# %%
# Converting the textual data to numerical data
vectorizer = TfidfVectorizer()
vectorizer.fit(X)

X = vectorizer.transform(X)
print(X)
# %%
#splitting the dataset to training and testing dataset
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, stratify=Y, random_state=2)


# %%
#training our model: Logistic Regression

model = LogisticRegression()
model.fit(X_train,Y_train)


# %%
#evaluation
#accuracy score on the training data

X_train_prediction = model.predict(X_train)

training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy score of the training data: ', training_data_accuracy)


# %%
#accuracy score on the test data
X_test_prediction = model.predict(X_test)
training_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score of the test data: ', training_data_accuracy)

# %%
# Making a predictive system
X_new = X_test[0]

prediction = model.predict(X_new)

# if(prediction[0] == 0):
#     print('The news is Real')
# else:
#     print('The news is Fake')


# %%
def preprocess_text(text):
    # Apply the same preprocessing (stemming, cleaning, etc.)
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [port_stem.stem(word) for word in text if word not in stopwords.words('english')]
    return ' '.join(text)

def predict_news(news_text, model, vectorizer):
    # Preprocess the text
    processed_text = preprocess_text(news_text)
    # Transform the text using the same vectorizer
    vectorized_text = vectorizer.transform([processed_text])
    # Predict using the trained model
    prediction = model.predict(vectorized_text)
    return prediction[0]


# %%
# Interactive loop for user input

user_input = input("Enter the news article (or type 'exit' to quit):\n")
if user_input.lower() == 'exit':
    print("Exiting the program.")
result = predict_news(user_input, model, vectorizer)
if result == 0:
    print("The news is likely REAL.")
else:
    print("The news is likely FAKE.")

# %%
