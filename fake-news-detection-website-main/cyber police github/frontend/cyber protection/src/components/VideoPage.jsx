import React, { useState } from "react";
import axios from "axios"; 
import "./FakeNewsDetector.css";
import "./VideoPage.css";
import DonutChart from "./chart";

export default function VideoPage() {
  const [article, setArticle] = useState("");
  const [image, setImage] = useState(null);
  const [articleResult, setArticleResult] = useState(null);
  const [articleResult1, setArticleResult1] = useState(null);
  const [imageResult, setImageResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [fakePercentage, setFakePercentage] = useState(0); // Store fake percentage in state
  const [realPercentage, setRealPercentage] = useState(0); // Store real percentage in state

  const imageSubmit = async () => {
    if (!image) {
      setImageResult("⚠️ Please upload a video before detecting.");
      return;
    }
    try {
      setFakePercentage(0);
      setRealPercentage(0);
      setLoading(true);
      setImageResult("");
      const formData = new FormData();
      formData.append("file", image); // Change "image" to "video"
    
      console.log("Sending FormData:", formData.get("file")); // Debugging
  
      let response = await fetch("http://127.0.0.1:8002/predict", {
        method: "POST",
        body: formData, // No headers needed for FormData
      });
  
      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status} - ${await response.text()}`);
      }
  
      let data = await response.json();
      console.log("API Response:", data);

      // Safely handle data fields to avoid runtime errors
      let manipulatedFrames = data.manipulated_frames || 0;
      let realFrames = data.real_frames || 0;
      let totalFrames = manipulatedFrames + realFrames;

      let calculatedFakePercentage = (manipulatedFrames / totalFrames) * 100;
      let calculatedRealPercentage = (realFrames / totalFrames) * 100;

      console.log(`Fake Percentage: ${calculatedFakePercentage.toFixed(2)}%`);
      console.log(`Real Percentage: ${calculatedRealPercentage.toFixed(2)}%`);

      // Set the state values for fake and real percentages
      setFakePercentage(calculatedFakePercentage);
      setRealPercentage(calculatedRealPercentage);
  
      setImageResult(data.result === "Manipulated" ? "⚠️ Fake Video Detected!" : "✅ Real Video Detected!");
    } catch (error) {
      console.error("Error sending video:", error.message);
      setImageResult("⚠️ API error. Please check input format.");
    } finally {
      setLoading(false);
    }
  };

  // Use state variables for data
  const data1 = [
    { name: "Real", value: realPercentage, color: "green" },
    { name: "Fake", value: fakePercentage, color: "Red" }
  ];

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;
  
    setImage(file);
    setImageResult("🔍 Analyzing Video...");
  };

  return (
    <div className="container1 dark-theme">
      <header className="header1">
        <h1 className="title1">📰 Fake News Detector</h1>
        <p className="subtitle1">AI-powered detection for video & images</p>
      </header>
      <div className="upload-box1">
        <p>Drag & drop your files here, paste, or <span className="browse1">browse</span></p>
      </div>
      <div className="gallery1">

        <div className="card1 image-section1 display1">
          <h2>🖼️ Upload Video</h2>
          <input type="file" accept="video/*" className="file-input1" onChange={handleImageUpload} />
          {image && <p className="image-name1">📌 {image.name}</p>}
          {imageResult && <p className="result1">{imageResult} Fake: {fakePercentage.toFixed(2)}% | Real: {realPercentage.toFixed(2)}%</p>}
          {loading ? <div className="single-loader"></div> : <button className="button1" onClick={imageSubmit}>Detect Fake Video</button>}
          {articleResult1 && <p className="result1"> {articleResult1} </p>}
        </div>

        {/* Render DonutChart only when percentages are available */}
        {(fakePercentage !== 0 || realPercentage !== 0) && <DonutChart text="Video Integrity" data={data1} />}
        
      </div>
    </div>
  );
}
