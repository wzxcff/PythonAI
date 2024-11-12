import tensorflow as tf
from keras import Sequential
from keras import layers
from keras import utils
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

# data = [
#     {"question": "How are you?", "answer": "Good, what about you?"},
#     {"question": "What's your name?", "answer": "My name is John."},
#     {"question": "What can you do?", "answer": "I can answer simple questions."},
#     {"question": "What's your house?", "answer": "My house is Snowy."},
#     {"question": "How old are you?", "answer": "99 years old"},
#     {"question": "I'm years old", "answer": "Ohhhh, that's great!"}
# ]

data = pd.read_csv("train.csv")


def tokenize_sentence(sentence):
    return word_tokenize(sentence.lower())


def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word not in stop_words]


stemmer = SnowballStemmer("english")


def stem_tokens(tokens):
    return [stemmer.stem(word) for word in tokens]


def preprocess_text(text):
    tokens = tokenize_sentence(text)
    tokens = remove_stopwords(tokens)
    tokens = stem_tokens(tokens)
    return " ".join(tokens)


questions = [preprocess_text(item) for item in data["question"]]
answers = data["answer"].tolist()

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(questions)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(answers)


Dense = layers.Dense
Dropout = layers.Dropout

model = Sequential([
    Dense(512, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.5),
    Dense(len(set(y)), activation='softmax')
])
y = utils.to_categorical(y)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X.toarray(), y, epochs=120, batch_size=2)

model.save("chatbot_python_model.h5")

def preprocess_query(query):
    query = preprocess_text(query)
    return vectorizer.transform([query]).toarray()


def get_response(query):
    processed_query = preprocess_query(query)
    prediction = model.predict(processed_query)
    predicted_class = np.argmax(prediction)
    return label_encoder.inverse_transform([predicted_class])[0]


while True:
    query = str(input("Ваш запрос: "))
    response = get_response(query)
    print(f"Ответ: {response}")
