from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import easyocr
import cv2
import numpy as np
import google.generativeai as genai
from transformers import pipeline, AutoTokenizer
from typing import List, Dict
import io
import os
from dotenv import load_dotenv
from PIL import Image

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)
# Load tokenizer for BART-based fake news detection
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli", use_fast=False)

# Load the fake news classifiers
bart_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
roberta_classifier = pipeline("text-classification", model="roberta-base-openai-detector")
electra_classifier = pipeline("text-classification", model="google/electra-large-discriminator")

reader = easyocr.Reader(['en'])

def perform_ocr(image: Image.Image) -> List[str]:
    """Extracts text from an image using OCR."""
    image = np.array(image)
    results = reader.readtext(image)
    return [text for (_, text, _) in results]

def classify_with_bart(text: str):
    candidate_labels = ["true", "fake"]
    classification = bart_classifier(text, candidate_labels)
    return classification["labels"][0], classification["scores"][0]

def classify_with_roberta(text: str):
    classification = roberta_classifier(text)
    return classification[0]["label"].lower(), classification[0]["score"]

def classify_with_electra(text: str):
    classification = electra_classifier(text)
    return classification[0]["label"].lower(), classification[0]["score"]

def classify_with_gemini(text: str) -> str:
    try:
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(f"Classify the following news text as 'true' or 'fake': {text}")
        
        if response and hasattr(response, 'text') and response.text:
            gemini_result = response.text.lower().strip()
            if "true" in gemini_result:
                return "true"
            elif "fake" in gemini_result:
                return "fake"
            else:
                return "unknown"
        return "error: Empty response from Gemini API"
    except Exception as e:
        error_message = str(e).lower()
        if "quota" in error_message or "429" in error_message:
            return "error: API quota exceeded. Try again later."
        return f"error: {error_message}"

def determine_final_result(text: str):
    bart_label, bart_score = classify_with_bart(text)
    roberta_label, roberta_score = classify_with_roberta(text)
    electra_label, electra_score = classify_with_electra(text)
    gemini_label = classify_with_gemini(text)
    
    labels = [bart_label, roberta_label, electra_label, gemini_label]
    labels = [label for label in labels if "error" not in label]
    
    if not labels:
        return "unknown", bart_score, roberta_score, electra_score
    
    final_label = max(set(labels), key=labels.count)  # Majority voting
    return final_label, bart_score, roberta_score, electra_score

@app.post("/analyze/")
async def upload_image(file: UploadFile = File(...)):
    """Endpoint to upload an image, perform OCR, and classify extracted text."""
    image = Image.open(file.file)
    texts = perform_ocr(image)
    if not texts:
        return {"message": "No text found in the image!"}
    
    true_count, false_count = 0, 0
    true_lines, false_lines = [], []
    bart_scores, roberta_scores, electra_scores = [], [], []
    
    for text in texts:
        label, bart_score, roberta_score, electra_score = determine_final_result(text)
        bart_scores.append(bart_score)
        roberta_scores.append(roberta_score)
        electra_scores.append(electra_score)
        
        if label == "true":
            true_count += 1
            true_lines.append(text)
        else:
            false_count += 1
            false_lines.append(text)
    
    final_result = "true" if true_count > false_count else "fake"
    avg_bart_score = sum(bart_scores) / len(bart_scores) if bart_scores else 0
    avg_roberta_score = sum(roberta_scores) / len(roberta_scores) if roberta_scores else 0
    avg_electra_score = sum(electra_scores) / len(electra_scores) if electra_scores else 0
    
    return {
        "true_count": true_count,
        "false_count": false_count,
        "true_lines": true_lines,
        "false_lines": false_lines,
        "final_decision": final_result,
        "average_confidence_scores": {
            "bart": avg_bart_score,
            "roberta": avg_roberta_score,
            "electra": avg_electra_score
        }
    }

   # Run the API (for local testing)
    if __name__ == "__main__":
        uvicorn.run(app, host="127.0.0.1", port=8001, reload=True)

    # uvicorn Finalmain:app --reload --port 8001
