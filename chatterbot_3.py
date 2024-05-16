import json
import numpy as np
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, QTextBrowser

# Step 1: Parse the JSON file
with open('intents.json', 'r') as file:
    intents = json.load(file)

# Step 2: Preprocess the data
patterns = []
responses = []
tags = []

for intent in intents['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        responses.append(intent['responses'])
        tags.append(intent['tag'])

# Step 3: Perform label encoding on the tags
label_encoder = LabelEncoder()
encoded_tags = label_encoder.fit_transform(tags)

# Step 4: Tokenize the patterns
tokenizer = Tokenizer()
tokenizer.fit_on_texts(patterns)
vocab_size = len(tokenizer.word_index) + 1

# Step 5: Convert text data to sequences
sequences = tokenizer.texts_to_sequences(patterns)
max_sequence_len = max([len(seq) for seq in sequences])
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_len, padding='post')

# Step 6: Build a neural network model
model = Sequential([
    Embedding(vocab_size, 64, input_length=max_sequence_len),
    LSTM(64),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(set(tags)), activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Step 7: Train the model
model.fit(padded_sequences, encoded_tags, epochs=500, batch_size=8)

# Step 8: Create the chatbot interface using PyQT5
class ChatBot(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle('ChatBot')
        self.setGeometry(100, 100, 400, 500)
        
        self.layout = QVBoxLayout()
        
        self.chat_display = QTextBrowser()
        self.layout.addWidget(self.chat_display)
        
        self.input_box = QLineEdit()
        self.input_box.returnPressed.connect(self.send_message)
        self.layout.addWidget(self.input_box)
        
        self.send_button = QPushButton('Send')
        self.send_button.clicked.connect(self.send_message)
        self.layout.addWidget(self.send_button)
        
        self.setLayout(self.layout)
    
    def send_message(self):
        user_input = self.input_box.text()
        self.input_box.clear()
        
        # Preprocess user input
        user_seq = tokenizer.texts_to_sequences([user_input])
        user_padded = pad_sequences(user_seq, maxlen=max_sequence_len, padding='post')
        
        # Predict intent
        prediction = model.predict(user_padded)
        predicted_tag = label_encoder.inverse_transform([np.argmax(prediction)])[0]
        
        # Get random response for the predicted tag
        responses_for_tag = [intent['responses'] for intent in intents['intents'] if intent['tag'] == predicted_tag]
        response = np.random.choice(responses_for_tag[0])
        
        # Display user input and bot response
        self.chat_display.append(f"You: {user_input}")
        self.chat_display.append(f"Bot: {response}")

if __name__ == '__main__':
    app = QApplication([])
    chatbot = ChatBot()
    chatbot.show()
    app.exec_()
