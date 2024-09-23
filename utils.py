import os

os.environ['TF_CPP_MIN_LOG_LEVE'] = '3'

import random
import pickle
import json

import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model
import logging


logger = logging.getLogger("Utils")
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def clean_up_sentence(sentence):

    logger.debug("Clean up sentence.")
    lemmatizer = WordNetLemmatizer()
    ignore_symbols = ['?', '!', '.', ',']

    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words if word not in ignore_symbols]

    return sentence_words


def bag_of_words(sentence):

    logger.debug("Load words.")
    words = pickle.load(open("model/words.pkl", "rb"))

    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)

    logger.debug("Create bag of words")
    for word in sentence_words:
        for index, sentence_word in enumerate(words):
            if sentence_word == word:
                bag[index] = 1

    return np.array(bag)


def predict_class(sentence):

    logger.debug("Load classes.")
    classes = pickle.load(open("model/classes.pkl", "rb"))

    logger.debug("Load model.")
    model = load_model("model/chatbot_model.keras")

    bow = bag_of_words(sentence)

    logger.info("Predict class.")
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25

    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)

    return_list = []

    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})

    return return_list


def get_response(intents_list):

    logger.info("Get response from intents list.")
    intents_json = json.load(open("dataset/intents.json"))

    tag = intents_list[0]["intent"]
    list_of_intents = intents_json["intents"]

    for i in list_of_intents:
        if i["tag"] == tag:
            result = random.choice(i["responses"])
            break

    return result
