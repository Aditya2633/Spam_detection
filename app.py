import streamlit as st
import pickle

# Load model and vectorizer
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("vectorize.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

# Streamlit UI
st.title("📩 Spam Message Detector")
st.markdown("Enter a message below and find out if it's **Spam** or **Not Spam**.")

# User input
user_input = st.text_area("✉️ Enter your message:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        # Optional: apply your custom preprocessing function here if used during training
         user_input_processed = user_input.lower()  # example if you lowercased during training

         input_vector = vectorizer.transform([user_input_processed])

         prediction = model.predict(input_vector)

         if prediction[0] == 0:
            st.error("🚫 This is a **Spam** message.")
         else:
            st.success("✅ This is **Not Spam**.")
