# train_model.py

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib

# Simulated 500 thoughts (expand or replace with actual thoughts)
thoughts = [
    "Good morning", "Good afternoon", "Have you eaten?", "Did you read?", "How are you?",
    "I feel sleepy", "I’m happy", "I’m tired", "Call mom", "Check messages",
    "Turn off the light", "What’s the time?", "Let's go", "Yes", "No", "Maybe", 
    "I need help", "I’m hungry", "I’m fine", "How's the weather?", "Have you worked out?",
    "Where are you?", "What’s for dinner?", "See you later", "Can you help me?", 
    "I’m busy", "Let’s meet", "I’m excited", "I’m bored", "What’s new?", "Do you want coffee?",
    "Let’s take a break", "Did you sleep well?", "Do you like coffee?", "Turn on the TV",
    "Send a message", "Buy groceries", "Watch a movie", "I love this song", "I miss you",
    "Happy Birthday!", "Do you need anything?", "Where’s my phone?", "I want to relax",
    "Let's go out", "I’m feeling good", "I’m working on something", "I can’t focus", 
    "Can you remind me?", "Tell me a joke", "How was your day?", "Read this article",
    "Drink water", "Stay positive", "Are you coming?", "I’m learning something new",
    "Check the news", "Go to the gym", "How was the meeting?", "I need a nap", "Can you do this for me?",
    "What's the plan?", "Can we talk?", "Don’t forget", "Call the doctor", "Where’s my charger?",
    "Let’s do this", "Is everything okay?", "Let’s go for a walk", "Get some fresh air", 
    "I’m so tired", "Let's play a game", "I’m going to bed", "How’s work going?", "Is it time yet?",
    "I need a coffee", "Have you heard this song?", "It’s too hot", "It’s too cold", "I’m feeling stressed",
    "Let’s chill", "Check your email", "What are you doing?", "I’ll be there in 10 minutes",
    "I’m getting ready", "I’m on my way", "How do you feel?", "Do you want to go out?", "I’ve finished",
    "You’re awesome", "Good job", "Thanks", "You're welcome", "I’m proud of you", "See you soon", 
    "I'm thinking", "I’m going to work", "Let's grab lunch", "Can you hear me?", "I love you", 
    "Take care", "Be careful", "Stay safe", "Don’t worry", "Enjoy your day", "Good night", 
    "I’m feeling great", "Let’s grab a drink", "It’s too late", "I’m looking forward to this",
    "Let's hang out", "How about tomorrow?", "I’m so happy", "It’s okay", "See you next time"
]  # You can expand this list to 500 thoughts

# Simulate EEG feature vectors (200 samples, 10 features each)
X = np.random.rand(200, 10)  # 200 EEG samples with 10 features each
y = np.random.choice(thoughts, size=200)  # Randomly assign thoughts to the samples

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Scale the features

# Train model (Support Vector Machine)
model = SVC(kernel='linear', probability=True)  # SVM with linear kernel
model.fit(X_scaled, y)  # Train the model on the EEG features and thought labels

# Save the trained model and scaler
joblib.dump(model, 'eeg_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model and scaler saved successfully.")
