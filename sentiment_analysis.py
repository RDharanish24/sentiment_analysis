

import os
import json
from zipfile import ZipFile
import pandas as pd
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam

kaggle_dictionary = json.load(open("kaggle.json"))

os.environ["KAGGLE_USERNAME"] = kaggle_dictionary["username"]
os.environ["KAGGLE_KEY"] = kaggle_dictionary["key"]

!kaggle datasets download lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

with ZipFile("imdb-dataset-of-50k-movie-reviews.zip", "r") as zip_ref:
    zip_ref.extractall()

data = pd.read_csv("IMDB Dataset.csv")

data.replace({"sentiment": {"positive": 1, "negative": 0}}, inplace=True)

train_data, test_data = train_test_split(
    data, test_size=0.2, random_state=42
)

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(train_data["review"])

x_train = pad_sequences(
    tokenizer.texts_to_sequences(train_data["review"]),
    maxlen=200
)

x_test = pad_sequences(
    tokenizer.texts_to_sequences(test_data["review"]),
    maxlen=200
)

y_train = train_data["sentiment"]
y_test = test_data["sentiment"]

model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128, input_length=200))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation="sigmoid"))

model.compile(
    optimizer=Adam(),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.2
)

loss,accuracy=model.evaluate(x_test,y_test)
print("Loss: ",loss)
print("Accuracy: ",accuracy)

def predict_sentiment(text):
  sequence = tokenizer.texts_to_sequences([text])
  padded = pad_sequences(sequence, maxlen=200)
  prediction = model.predict(padded)
  sentiment="positive" if prediction[0][0]>0.5 else "negative"
  return sentiment

review="i do not like this movie"
sentiment=predict_sentiment(review)
print(f"The sentiment of the review is {sentiment}")

