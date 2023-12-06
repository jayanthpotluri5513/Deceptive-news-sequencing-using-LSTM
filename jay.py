
from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
import numpy as np
from nltk.corpus import stopwords
nltk.download('stopwords')
app = Flask(__name__)
tf1 = TfidfVectorizer(stop_words='english', max_df=0.7, max_features=5000)
loaded_model = pickle.load(open('model_news2.pkl', 'rb'))
df = pd.read_csv('fakenews.csv')
df=df.drop(columns=['id','title','author'],axis=1)
df=df.dropna(axis=0) # Removing empty fields from dataset
X = df['text']
y = df['label']
df['clean_news']=df['text'].str.lower() ## converting uppercase to lowercase
df['clean_news'] = df['clean_news'].str.replace('\n', '')
df['clean_news'] = df['clean_news'].str.replace('\s+', ' ')   ## Replacing all special characters, empty/null characters from the dataset
def remove_special_characters(text):
    return re.sub(r'[^A-Za-z0-9\s]', '', text)

df['clean_news'] = df['clean_news'].apply(remove_special_characters)
stop=stopwords.words('english')
df['clean_news']=df['clean_news'].apply(lambda x:" ".join([word for word in x.split() if word not in stop]))
X=tf1.fit_transform(df['clean_news']).toarray()
y=df['label'] 
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def fake_news_det(news):
    vectorized_input_data = tf1.transform([news])
    prediction = loaded_model.predict(vectorized_input_data)
    return prediction[0]
@app.route('/')
def home():
    print("Home route accessed")
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        pred = fake_news_det(message)
        print(pred)
        return render_template('index.html',prediction=pred)
    else:
        return render_template('index.html', prediction="Something went wrong")

if __name__ == '__main__':
   app.run(debug=True, port=5002)