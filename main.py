import json
import joblib
import os
import time
import numpy as np
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INTENT_MODEL_PATH = os.path.join(BASE_DIR, "joblib", "intent_model.joblib")
INTENT_VECTORIZER_PATH = os.path.join(BASE_DIR, "joblib", "intent_vectorizer.joblib")
RESPONSES_PATH = os.path.join(BASE_DIR, "response", "responses.json")
HOTEL_DATA_PATH = os.path.join(BASE_DIR, "response", "hotel-data.json")

PROBABILITY_THRESHOLD = 0.45

#! Load model + vectorizer
try:
    model = joblib.load(INTENT_MODEL_PATH)
    vectorizer = joblib.load(INTENT_VECTORIZER_PATH)
except FileNotFoundError:
    st.error("Model or vectorizer file not found.")
    st.stop()

#! Load json
try:
    with open(RESPONSES_PATH, "r") as f:
        responses = json.load(f)
except FileNotFoundError:
    st.error("Responses file not found.")
    st.stop()

try:
    with open(HOTEL_DATA_PATH, "r") as f:
        hotel_data = json.load(f)
except FileNotFoundError:
    st.error("Hotel data file not found.")
    st.stop()

#! Chatbot reply function
def predict_intent(user_input):
    user_input = user_input.lower()

    start_time = time.time()

    vec = vectorizer.transform([user_input])
    intent = model.predict(vec)[0]

    # confidence score
    proba = model.predict_proba(vec)[0]
    confidence = float(np.max(proba))
    max_proba = proba.max()
    if max_proba < PROBABILITY_THRESHOLD:
        intent = "unknown_intent"

    # chatbot response
    response = responses.get(intent, responses.get("unknown_intent"))
    if isinstance(response, list):
        response = response[0]

    elapsed = round((time.time() - start_time) * 1000, 2)

    return {
        "intent": intent,
        "confidence": confidence,
        "response": response.format(**hotel_data),
        "time_ms": elapsed
    }

#! Streamlit app
def main():
    st.set_page_config(page_title="Hotel FAQ Chatbot", layout="centered")
    st.title("Astra Imperium FAQ Chatbot (Logistic Regression)")
    st.caption("Ask me about room prices, check-in times, or facilities!")

    # initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you with your hotel booking today?"}]

    # display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # handle user input
    if prompt := st.chat_input("Type your message here..."):
        # display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        responses = predict_intent(prompt)

        # display chatbot reply
        st.session_state.messages.append({"role": "assistant", "content": responses['response']})
        with st.chat_message("assistant"):
            st.markdown(responses['response'])
            st.caption(
                f"**Predicted Intent:** `{responses['intent']}` | "
                f"**Confidence:** {responses['confidence']*100:.2f}% | "
                f"**Time Taken:** {responses['time_ms']} ms"
            )
    
if __name__ == '__main__':
    main()
