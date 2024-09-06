from transformers import pipeline
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()
MODEL_NAME = os.getenv("MODEL_NAME")
TASK = os.getenv("TASK")
GEMINI_APIKEY = os.getenv("GEMINI_APIKEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL")

class DataClassifier:
  def __init__(self, model_name = MODEL_NAME,
               device = None):
    self.model_name = model_name
    self.device = device
    self.task = TASK
    self.classifier = pipeline(self.task, model=model_name, device=self.device)
    self.gemini_name = GEMINI_MODEL
    api_key = GEMINI_APIKEY
    genai.configure(api_key=api_key)

    # safety setting
    safety_setting = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE"
        },
    ]

    # llm prediction
    self.llm = genai.GenerativeModel(self.gemini_name, safety_settings=safety_setting)

  def extract_aspect(self, sentence):
    # candidate labels
    labels = ["Price fairness", "Cleanliness", "Facilities", "Staff quality", "Convenience",
              "Punctuality", "Accessibility", "Safety", "Data availability"]

    # get result
    aspect = self.classifier(sentence, labels)
    return aspect

  def get_sentiment_based_on_aspect(self, sentence, aspect):
    labels = ["Positive", "Negative", "Neutral"]
    result = self.classifier(sentence, labels)
    return result

  def llm_prediction(self, prompt):
    try:
      response = self.llm.generate_content(prompt)
      return response.text
    except Exception as e:
      print(e)
      raise ValueError(e)

  def classifier_pipeline(self, review):
      # Extract aspects and their scores from the review
      r1 = self.extract_aspect(review)
      aspects = r1["labels"]
      scores = r1["scores"]

      # Map aspect to its score
      aspect_score = dict(zip(aspects, scores))

      # Apply a threshold to filter aspects
      threshold = 0.22
      aspect_score = {k: v for k, v in aspect_score.items() if v > threshold}

      if not aspect_score:
          return {"aspect": "None", "sentiment": "Neutral", "opinion": "No prominent aspect identified"}

      # Get the most prominent aspect
      most_prominent_aspect = max(aspect_score, key=aspect_score.get)

      # Create a prompt for LLM to generate an opinion based on the most prominent aspect
      prompt = f"""Get the opinion of user in this review: {review} that corresponded to aspect: {most_prominent_aspect}
      ***EXAMPLE***
      review example: 'I am impressed about the cleanliness, facility at station is really good'
      output: 'Impressed about cleanliness'
      ***NOTE***
      Do not have any markup syntax used in Markdown.
      Do not have symbol '\n' and the end or start
      Do not summarize the review
      The output must be original do not change the text
      """

      # Generate the opinion using the LLM
      opinion = self.llm_prediction(prompt).replace("\n", "").strip()

      # Get sentiment based on the most prominent aspect and generated opinion
      result = self.get_sentiment_based_on_aspect(opinion, most_prominent_aspect)

      # Map sentiment to score
      sentiments = result["labels"]
      scores = result["scores"]
      sentiment_score = dict(zip(sentiments, scores))

      # Determine the sentiment with the highest score
      sentiment = max(sentiment_score, key=sentiment_score.get)

      # Return the result as a dictionary
      return {"aspect": most_prominent_aspect, "sentiment": sentiment, "opinion": opinion}