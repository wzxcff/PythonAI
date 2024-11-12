import telebot
from dotenv import load_dotenv
from os import getenv
from keras import models
from nltk import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import pickle


load_dotenv()
bot = telebot.TeleBot(getenv('TOKEN'))
model = models.load_model("chatbot_python_model.keras")
stemmer = SnowballStemmer("english")

with open("vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

with open("label_encoder.pkl", "rb") as file:
    label_encoder = pickle.load(file)


def tokenize_sentence(sentence):
    return word_tokenize(sentence.lower())


def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word not in stop_words]


def stem_tokens(tokens):
    return [stemmer.stem(word) for word in tokens]


def preprocess_text(text):
    tokens = tokenize_sentence(text)
    tokens = remove_stopwords(tokens)
    tokens = stem_tokens(tokens)
    return " ".join(tokens)


def preprocess_query(query):
    query = preprocess_text(query)
    return vectorizer.transform([query]).toarray()


def get_response(query):
    processed_query = preprocess_query(query)
    prediction = model.predict(processed_query)
    predicted_class = np.argmax(prediction)
    return label_encoder.inverse_transform([predicted_class])[0]


@bot.message_handler(content_types=['text'])
def message_handler(message):
    if message.text == "/start":
        bot.send_message(message.chat.id, "Hello! Ask me something")
    else:
        response = get_response(message.text)
        bot.send_message(message.chat.id, response)


bot.polling(none_stop=True)
