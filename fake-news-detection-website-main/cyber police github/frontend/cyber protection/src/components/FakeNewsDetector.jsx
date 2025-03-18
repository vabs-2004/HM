import React, { useState } from "react";
import axios from "axios";
import "./FakeNewsDetector.css";
import DonutChart from "./chart";

export default function FakeNewsDetector() {
  const [article, setArticle] = useState("");
  const [image, setImage] = useState(null);
  const [articleResult, setArticleResult] = useState(null);
  const [imageResult, setImageResult] = useState(null);
  const [articleConfidence, setArticleConfidence] = useState(null);
  const [imageConfidence, setImageConfidence] = useState(null);
  const [loading, setLoading] = useState(false);
  const [loading2, setLoading2] = useState(false);
  const [article2, setArticle2] = useState("");
  const [article1, setArticle1] = useState("");
  const [article3, setArticle3] = useState("");
  const [article4, setArticle4] = useState("");

  // Send article for classification
  const sendData = async () => {
    if (!article.trim()) {
      setArticleResult("⚠️ Please enter an article before detecting.");
      return;
    }

    const requestData = {
      article_text: article,
    };
    setLoading(true);
    try {
      setArticleResult(null)
      setArticleConfidence(null)
      const response = await axios.post(
        "http://127.0.0.1:8000/classify_news",
        requestData,
        {
          headers: { 
            "Content-Type": "application/json",
            "Accept": "application/json"
          },
        }
      );

      console.log("Server Response:", response.data);

      const result = response.data.External_fact_verification === "Fake News" ? "⚠️ Fake News Detected!" : "✅ Real News Detected!";
      setArticleResult(result);

      let final1 = parseFloat(response.data.final_confidence) * 100;
      setArticleConfidence(final1); // Store it as a number, not a string
      console.log(articleConfidence);

      if (response.data.External_fact_verification === "Fake News") {
        setArticle2("Real News");
        setArticle1("Fake News");
      } else {
        setArticle2("Fake News");
        setArticle1("Real News");
      }
    } catch (error) {
      console.error("Error sending data:", error.response?.data || error.message);
      setArticleResult("⚠️ API error. Please check input format.");
      setArticleConfidence(null);
    } finally {
      setLoading(false);
    }
  };

  // Submit image for analysis
  const imageSubmit = async () => {
    if (!image) {
      setImageResult("⚠️ Please upload an image before detecting.");
      return;
    }
    try {
      setImageConfidence(null);
      setImageResult(null);
      setLoading2(true);
      const formData = new FormData();
      formData.append("file", image);

      let response = await fetch("http://127.0.0.1:8001/analyze/", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status} - ${await response.text()}`);
      }

      let data = await response.json();
      console.log("API Response:", data);

      setImageResult(data.final_decision === "fake" ? "⚠️ Fake News Detected!" : "✅ Real News Detected!");
      const sum = (data.average_confidence_scores.bart || 0) +
        (data.average_confidence_scores.roberta || 0) +
        (data.average_confidence_scores.electra || 0);
      const average = sum / 3;
      let final = average * 100;
      setImageConfidence(final); // Store it as a number

      if (data.final_decision === "fake") {
        setArticle3("Real News");
        setArticle4("Fake News");
      } else {
        setArticle3("Fake News");
        setArticle4("Real News");
      }
    } catch (error) {
      console.error("Error sending image:", error.message);
      setImageResult("⚠️ API error. Please check input format.");
      setImageConfidence(null);
    } finally {
      setLoading2(false);
    }
  };

  // Handle image upload
  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setImage(file);
    setImageResult("🔍 Analyzing Image...");
  };

  // Prepare data for DonutChart (based on confidence values)
  const articleData = [
    { name: article1, value: articleConfidence || 0, color: "green" },
    { name: article2, value: 100 - (articleConfidence || 0), color: "red" }
  ];

  const imageData = [
    { name: article3, value: imageConfidence || 0, color: "green" },
    { name: article4, value: 100 - (imageConfidence || 0), color: "red" }
  ];

  return (
    <div className="container1 dark-theme">
      <header className="header1">
        <h1 className="title1">📰 Fake News Detector</h1>
        <p className="subtitle1">AI-powered detection for text & images</p>
      </header>

      <div className="gallery1">
        {/* ARTICLE DETECTION SECTION */}
        <div className="card1 article-section1">
          <h2>📄 Enter the Text Article</h2>
          <textarea
            placeholder="Paste article here..."
            value={article}
            onChange={(e) => setArticle(e.target.value)}
            className="textarea1"
          />
          {loading ? <div className="single-loader"></div> : <button className="button1" onClick={sendData}>Detect Fake News</button>}
          {articleResult && <p className="result1">{articleResult}</p>}

          {/* Display DonutChart for article confidence */}
          {articleConfidence !== null && (
            <DonutChart text="Article Confidence" data={articleData} />
          )}
        </div>

        {/* IMAGE DETECTION SECTION */}
        <div className="card1 image-section1 display1">
          <h2>🖼️ Upload Image for Analysis</h2>
          <input type="file" className="file-input1" onChange={handleImageUpload} />
          {image && <p className="image-name1">📌 {image.name}</p>}
          
          {loading2 ? <div className="single-loader"></div> : <button className="button1" onClick={imageSubmit}>Detect Fake Image</button>}
          {imageResult && <p className="result1">{imageResult}</p>}

          {/* Display DonutChart for image confidence */}
          {imageConfidence !== null && (
            <DonutChart text="Image Confidence" data={imageData} />
          )}
        </div>
      </div>
    </div>
  );
}
