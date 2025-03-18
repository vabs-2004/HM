from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from transformers import pipeline
from typing import Dict, List
from huggingface_hub import HfFolder
import google.generativeai as genai
import os
from fastapi.middleware.cors import CORSMiddleware
import csv
import pandas as pd
from dotenv import load_dotenv
import re
import nltk
import uvicorn
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  
)

# Set your Hugging Face token
HF_TOKEN = os.getenv("HF_TOKEN")
HfFolder.save_token(HF_TOKEN)


#model_name = "cross-encoder/nli-distilroberta-base"

# Load tokenizer and model for multilingual classification
#model_name = "joeddav/xlm-roberta-large-xnli"  # ✅ Supports 100+ languages
#tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

# Zero-shot classification pipeline with XLM-RoBERTa
#fake_news_classifier = pipeline("zero-shot-classification", model=model_name, tokenizer=tokenizer)

# CSV File Path
CSV_FILE = "news_data.csv"

# Ensure CSV exists with only "Article" and "Label" headers
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Article", "Label"])  # Only store Article and Label

GENAI_API_KEY = os.getenv("GENAI_API_KEY")
if not GENAI_API_KEY:
    raise ValueError("GENAI_API_KEY is not set in environment variables")
genai.configure(api_key=GENAI_API_KEY)
model_gemini = genai.GenerativeModel('gemini-2.0-flash')

# Pydantic model for request
class NewsRequest(BaseModel):
    article_text: str

# Load Zero-Shot Classification Models
models = {
    "DeBERTa": pipeline("zero-shot-classification", model="microsoft/deberta-v3-large"),
    "DistilBERT": pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli"),
    "BART": pipeline("zero-shot-classification", model="facebook/bart-large-mnli"),
}

def classify_news_ensemble(article_text):
    labels = ["Fake News", "Real News"]
    scores = {"Fake News": [], "Real News": []}
    model_results = []

    for name, classifier in models.items():
        try:
            result = classifier(article_text, candidate_labels=labels, truncation=True, max_length=512)
            label = result["labels"][0]
            confidence = result["scores"][0]
            model_results.append({"model": name, "label": label, "confidence": confidence})
            scores[label].append(confidence)
        except Exception as e:
            print(f"⚠️ Error with model {name}: {e}")

    avg_scores = {label: sum(confidences) / len(confidences) for label, confidences in scores.items() if confidences}

    if not avg_scores:
        return "Unable to classify", 0.0, model_results

    final_label = max(avg_scores, key=avg_scores.get)
    final_confidence = avg_scores[final_label]

    return final_label, final_confidence, model_results

# Download required resources (only once)
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    return ' '.join(
        lemmatizer.lemmatize(word)
        for word in re.sub(r'<.*?>|http\S+|www\S+|[^a-zA-Z\s]', '', text.lower()).split()
        if word not in stop_words
    )

def classify_news_geminimodel(article_text):
    """Classifies news using Google Gemini API."""
    prompt = f"""
    Classify the following news article as either 'Fake News' or 'Real News'.
    Only return one of these two labels.
    Article: "{article_text}"
    """
    response = model_gemini.generate_content(prompt) # Use generate_content instead of generate_text
   
    # Return the stripped classification result
    return response.text.strip()

class ArticleRequest(BaseModel):
    article_text: str

class ModelResult(BaseModel):
    model: str
    label: str
    confidence: float

class ClassificationResponse(BaseModel):
    External_fact_verification: str
    Models_final_prediction: str
    final_confidence: float
    model_results: List[ModelResult]

@app.post("/classify_news", response_model=ClassificationResponse)
def classify_news(article: ArticleRequest):
    processed_text = preprocess_text(article.article_text)
    final_label, final_confidence, model_results = classify_news_ensemble(processed_text)

    # Fact verification using Gemini
    fact_verification = classify_news_geminimodel(article.article_text)

    # Save result to CSV
    with open(CSV_FILE, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([article.article_text, fact_verification])

    return ClassificationResponse(
        External_fact_verification=fact_verification,
        Models_final_prediction=final_label,
        final_confidence=round(final_confidence, 2),
        model_results=model_results
    )

@app.get("/dataset")
def get_dataset():
    """Returns the stored dataset as JSON."""
    if not os.path.exists(CSV_FILE):
        raise HTTPException(status_code=404, detail="No dataset found.")
    
    df = pd.read_csv(CSV_FILE)
    return df.to_dict(orient="records")

@app.get("/download_dataset")
def download_dataset():
    if os.path.exists(CSV_FILE):
        return FileResponse(CSV_FILE, media_type="text/csv", filename="news_dataset.csv")
    return {"error": "Dataset not found"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
    
# uvicorn Fakenews:app --reload --port 8000