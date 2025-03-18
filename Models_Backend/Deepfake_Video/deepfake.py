from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2
import timm
import numpy as np
import tempfile
import shutil
import os
import uvicorn
import asyncio

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=2)
model.load_state_dict(torch.load("best_vit_model.pth", map_location=device))
model.to(device).eval().half() 

# Function to detect significant scene changes
def is_significant_change(prev_frame, current_frame, threshold=30):
    diff = cv2.absdiff(prev_frame, current_frame)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    score = cv2.mean(gray_diff)[0]
    return score > threshold

# Function to process the video and classify frames
def predict_deepfake(video_path, model, transform, device, frame_skip=5, batch_size=8):
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)  # Here,I use Video Decoding
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # Here,I Reduce buffering delay
    frame_count = 0
    real_count = 0
    manipulated_count = 0
    prev_frame = None
    batch = []
    predictions = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Here, I process key frames only (scene changes)
        if prev_frame is None or frame_count % frame_skip == 0 or is_significant_change(prev_frame, frame):
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image = transform(image).unsqueeze(0).to(device).half()  # Here, I convert input to FP16
            batch.append(image)

            # Here, I process batch when it reaches batch_size
            if len(batch) == batch_size:
                batch_tensor = torch.cat(batch, dim=0)
                with torch.no_grad():
                    outputs = model(batch_tensor)
                    predicted = torch.argmax(outputs, dim=1).tolist()
                predictions.extend(predicted)
                batch = []
        
        prev_frame = frame  # Here, i store previous frame for scene change detection

    # Here, i process remaining frames in batch
    if batch:
        batch_tensor = torch.cat(batch, dim=0)
        with torch.no_grad():
            outputs = model(batch_tensor)
            predicted = torch.argmax(outputs, dim=1).tolist()
        predictions.extend(predicted)

    cap.release()
    os.remove(video_path)  # Here, I remove temp file

    # Count real vs manipulated frames
    real_count = predictions.count(0)
    manipulated_count = predictions.count(1)
    result = "Real" if real_count > manipulated_count else "Manipulated"
    return {"result": result, "real_frames": real_count, "manipulated_frames": manipulated_count}

#Asynchronous file deletion to avoid blocking response time
async def delete_temp_file(path):
    await asyncio.sleep(1)  #Ensure file is not in use before deletion
    os.remove(path)

@app.post("/predict")
async def predict_deepfake_api(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        shutil.copyfileobj(file.file, temp_video)
        temp_video_path = temp_video.name
    
    #video processing in a background thread
    result = await asyncio.to_thread(predict_deepfake, temp_video_path, model, transform, device)

    #Delete temp file asynchronously
    asyncio.create_task(delete_temp_file(temp_video_path))

    return result

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8002, reload=True)
