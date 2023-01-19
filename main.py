#imports
import discord
import os
import json
import random
import pickle
import numpy as np
import nltk
#specific from library imports
from keras.models import load_model
from keras.models import Sequential
from nltk.stem import WordNetLemmatizer
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
from dotenv import load_dotenv

#neural network lemmatizer
lemmatizer = WordNetLemmatizer()

#read intents.json
intents = json.loads(open('intents.json').read())

# creating empty lists to store data
words = []
classes = []
documents = []
ignore_letters = ["?", "!", ".", ","]
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # separating words from patterns
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)  # and adding them to words list
          
        # associating patterns with respective tags
        documents.append(((word_list), intent['tag']))
  
        # appending the tags to the class list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

#store the root words
words = [lemmatizer.lemmatize(word)
        for word in words if word not in ignore_letters]
words = sorted(set(words))

# saving the words and classes list to binary files
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

#changes bag of words to binary for bot
training = []
output_empty = [0]*len(classes)
for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(
        word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
          
    # making a copy of the output_empty
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])
random.shuffle(training)
training = np.array(training)

#split data
train_x = list(training[:, 0])
train_y = list(training[:, 1])

#creating sequential training model for neural network
print("Creating model!")
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]), ),
                activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), 
                activation='softmax'))
  
# compiling the model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd, metrics=['accuracy'])
hist = model.fit(np.array(train_x), np.array(train_y),
                 epochs=200, batch_size=5, verbose=1)
  
# saving the model
model.save("chatbotmodel.h5", hist)
print("Model has been created")

#loading the files created by neural network
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbotmodel.h5')

#cleans up input
def clean_up_sentences(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) 
                      for word in sentence_words]
    return sentence_words

#bag of words function
def bagw(sentence):
    # separate out words from the input sentence
    sentence_words = clean_up_sentences(sentence)
    bag = [0]*len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
  
            # check whether the word
            # is present in the input as well
            if word == w:
  
                # as the list of words
                # created earlier.
                bag[i] = 1
  
    # return a numpy array
    return np.array(bag)

#predict class function
def predict_class(sentence):
    bow = bagw(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) 
               if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]],
                            'probability': str(r[1])})
        return return_list

#function to create response
def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    result = ""
    for i in list_of_intents:
        if i['tag'] == tag:
            
              # prints a random response
            result = random.choice(i['responses'])  
            break
    return result
  
print("Neural network has been loaded successfully!")

#establishes client
client = discord.Client()

#logs in
print("Attempting login")
@client.event
async def on_ready():
    print('We have logged in as {0.user}'.format(client))

#on message
@client.event
async def on_message(message):
    if message.author == client.user:
        return

    author = str(message.author)

    if message.content.startswith('$'):

        if message.content.startswith("$hello"):
            print("user typed hello")
            await message.channel.send("Hello "+str(author))
            return

        #neural network inputs
        ints = predict_class(message.content)
        res = get_response(ints, intents)
        await message.channel.send(res)

#runs token to connect to client
client.run('ODc0MDg1ODI3MTE2OTI5MDM1.GZr-Xi.UBrVy7iWVF23HBwIPlQk6p2cy-LClRTJ0d6M64')



       

