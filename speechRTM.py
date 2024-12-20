import faiss
import numpy as np
import pandas as pd
import speech_recognition as sr
import spacy
from sentence_transformers import SentenceTransformer, util
from gtts import gTTS
import os

# Load necessary tools
recognizer = sr.Recognizer()
nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer('all-MiniLM-L6-v2')  # Small, fast embedding model

# Step 1: Dataset and FAISS Index Setup
print("Preparing dataset and FAISS index...")
data = pd.DataFrame({
    'content': [
        "The Q1 budget is $50,000.",
        "The project deadline is March 31st.",
        "Marketing needs more resources.",
        "We expect revenue growth of 15% next quarter.",
        "The hiring process will start in April."
    ]
})

# Generate embeddings for the dataset
embeddings = np.vstack([model.encode(row) for row in data['content']])

# Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity search
index.add(embeddings)  # Add dataset embeddings to the FAISS index

print("FAISS index ready. System is listening for questions...")

# Step 2: Speech-to-Text
def speech_to_text():
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=.5)  # Adjust for ambient noise
        print("Listening for input...")
        try:
            audio = recognizer.listen(source, phrase_time_limit=10, timeout=10)  # Longer phrase time limit and timeout
            return recognizer.recognize_google(audio)  # Use show_all=False for faster response
        except sr.WaitTimeoutError:
            print("Listening timed out while waiting for phrase to start")
            return ""
        except sr.UnknownValueError:
            return "Sorry, I could not understand the audio."
        except sr.RequestError as e:
            return f"Could not request results; {e}"

# Step 3: Detect if Input is a Question
def is_question(text):
    doc = nlp(text)
    return any(token.tag_ in ['WP', 'WRB'] or token.text == '?' for token in doc)

# Step 4: Query FAISS Index for Best Match
def query_faiss(text):
    text_embedding = model.encode(text).reshape(1, -1)
    distances, indices = index.search(text_embedding, k=1)  # Top 1 result
    if distances[0][0] < 1.8:  # Threshold to ensure relevance
        return data['content'][indices[0][0]]
    else:
        return "Sorry, I couldn't find any relevant information."

# Step 5: Text-to-Speech Output
def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    tts.save("response.mp3")
    os.system("start response.mp3")  # Use 'start' for playback on Windows

# Step 6: Process Input
def process_input(user_input):
    if "Sorry, I could not understand the audio." in user_input or "Could not request results" in user_input:
        print(user_input)
    else:
        response = query_faiss(user_input)
        print(f"OUT: {response}")

# Step 7: Real-Time Monitoring Loop
if __name__ == "__main__":
    print("System is live. Speak to ask a question or make a statement...")
    while True:
        user_input = speech_to_text()
        if user_input:
            print(f"User: {user_input}")
            process_input(user_input)