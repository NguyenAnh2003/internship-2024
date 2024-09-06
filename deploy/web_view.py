import streamlit as st
from transformers import pipeline

# Replace this with the path to your DataClassifier module
from classifier import DataClassifier
import torch
from dotenv import load_dotenv
import os

load_dotenv()

GEMINI_APIKEY = os.getenv("GEMINI_APIKEY")
MODEL_NAME = os.getenv("MODEL_NAME")
TASK = os.getenv("TASK")

# Initialize the DataClassifier instance
# Make sure to replace 'YOUR_API_KEY' with your actual Gemini API key
device = "cuda" if torch.cuda.is_available() else "cpu"
classifier = DataClassifier(model_name=MODEL_NAME, device=device)

# Streamlit app
def main():
    st.title("Review Analyzer")

    # Input text box for user to enter review
    review = st.text_area("Enter a review:", height=150)

    if st.button("Analyze"):
        if review:
            try:
                # Analyze the review using the classifier pipeline
                result = classifier.classifier_pipeline(review)

                # Display the results
                st.subheader("Analysis Result")
                st.write(f"**Aspect:** {result['aspect']}")
                st.write(f"**Sentiment:** {result['sentiment']}")
                st.write(f"**Opinion:** {result['opinion']}")
                st.write("---")

            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter a review to analyze.")

if __name__ == "__main__":
    main()
