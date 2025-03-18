import os
import easyocr
import cv2
import matplotlib.pyplot as plt
import numpy as np
import uvicorn
import google.generativeai as genai
from fastapi import FastAPI, File, UploadFile
from transformers import pipeline, AutoTokenizer
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Load tokenizer for DistilBART-based fake news detection
model_name = "valhalla/distilbart-mnli-12-1"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

# Load the fake news classifier (DistilBART)
fake_news_classifier = pipeline(
    "zero-shot-classification",
    model=model_name
)

# OCR function to extract text from images
def perform_ocr(image: Image.Image):
    """Performs OCR on an image and extracts text line by line."""
    reader = easyocr.Reader(['en'])
    image = np.array(image)
    results = reader.readtext(image)

    extracted_texts = []
    for (bbox, text, _) in results:
        extracted_texts.append(text)

    # No visualization or pop-up image display
    return extracted_texts

# Function to classify text using DistilBART
def classify_with_distilbart(text: str):
    candidate_labels = ["true", "fake"]
    classification = fake_news_classifier(text, candidate_labels)
    return classification["labels"][0]  # "true" or "fake"

# Function to classify fake news from extracted texts
def classify_fake_news(texts):
    """Classifies each line as true or fake and tracks the counts."""
    true_count, false_count = 0, 0
    true_lines, false_lines = [], []

    for text in texts:
        label = classify_with_distilbart(text)
        if label == "true":
            true_count += 1
            true_lines.append(text)
        else:
            false_count += 1
            false_lines.append(text)

    return true_count, false_count, true_lines, false_lines

# Function to classify text using Gemini API
def classify_with_gemini(text: str):
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

# Function to determine final classification
def determine_final_result(text: str):
    bart_label = classify_with_distilbart(text)
    gemini_label = classify_with_gemini(text)

    if "error" in gemini_label:
        return bart_label

    return bart_label if bart_label == gemini_label else gemini_label

# API endpoint to analyze news from an image
@app.post("/analyze/")
async def analyze_image(file: UploadFile = File(...)):
    try:
        if not file:
            return {"error": "No file uploaded!"}

        # Read image
        image = Image.open(file.file)

        # Extract text using OCR
        extracted_text = perform_ocr(image)
        if not extracted_text:
            return {"error": "No readable text found in image!"}

        # Classify extracted text
        true_count, false_count, true_lines, false_lines = classify_fake_news(extracted_text)
        final_result = "true" if true_count > false_count else "fake"

        return {
            "filename": file.filename,
            "content_type": file.content_type,
            "extracted_text": extracted_text,
            "true_count": true_count,
            "false_count": false_count,
            "true_lines": true_lines,
            "false_lines": false_lines,
            "final_decision": final_result
        }

    except Exception as e:
        return {"error": f"Failed to process image: {str(e)}"}

# Run the API (for local testing)
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001, reload=True)

# uvicorn newmain:app --reload --port 8001