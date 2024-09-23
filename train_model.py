import os

os.environ['TF_CPP_MIN_LOG_LEVE'] = '3'

import random
import pickle
import json

import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

lemmatizer = WordNetLemmatizer()

intents = json.load(open('dataset/intents.json'))

words = []
classes = []
documents = []
ignore_symbols = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern) # Tokenizes each pattern sentence into words.
        words.extend(word_list)
        documents.append((word_list, intent['tag'])) # Associates each tokenized pattern with its corresponding intent tag.
        if intent['tag'] not in classes:
            classes.append(intent['tag'])


words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_symbols] # Convert words to lowercase and reduces them to their base form and exclude punctuation marks.
words = sorted(set(words)) # eliminate duplicates and sort the list of words

classes = sorted(set(classes))

pickle.dump(words, open('model/words.pkl', 'wb'))
pickle.dump(classes, open('model/classes.pkl', 'wb'))

training = []
output_empty = [0] * len(classes) # A list of zeros with a length equal to the number of classes, used as a template for one-hot encoding.


for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word) for word in word_patterns if word not in ignore_symbols]

    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0) # Create a binary vector (bag of words) where each position corresponds to a word in the vocabulary.

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row]) # Each training example consists of the bag of words and the one-hot encoded intent class.

random.shuffle(training)
# training = np.array(training)
training = np.array(training, dtype=object)

train_x = list(training[:, 0]) # Input features (Bag of words)
train_y = list(training[:, 1]) # Output labels (one-hot encoded classes)

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(lr=0.01, weight_decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

model.save('model/chatbot_model.keras')