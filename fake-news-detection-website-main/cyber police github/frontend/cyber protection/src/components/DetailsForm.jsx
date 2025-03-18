import React from "react";
import "./DetailsForm.css";
import { useState } from "react";

export default function DetailsForm() {

  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const[phone, setPhone] = useState("");
  const[message, setMessage] = useState("");

  const sendData = async (e) => {
    e.preventDefault(); // Prevent page reload
    try {
      const res = await fetch("http://localhost:5000/details/form", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          name,
          email,
          phone,
          message,
        }),
      });

      const data = await res.json();
      console.log("Signup Response:", data);
      alert("Signup Successful!"); // Notify the user
    } catch (error) {
      console.error("Error:", error);
      alert("Signup Failed!");
    }
    }

  return (
    <div className="form-container">
      <div className="form-card">
        <h2 className="form-title">Enter Your Details</h2>
        <form className="details-form" onSubmit={sendData}>
          <div className="input-group">
            <label htmlFor="name">Full Name</label>
            <input type="text" id="name" placeholder="Enter your full name" onChange={(e) => setName(e.target.value)} required />
          </div>

          <div className="input-group">
            <label htmlFor="email">Email Address</label>
            <input type="email" id="email" placeholder="Enter your email" onChange={(e) => setEmail(e.target.value)} required />
          </div>

          <div className="input-group">
            <label htmlFor="phone">Phone Number</label>
            <input type="tel" id="phone" placeholder="Enter your phone number" onChange={(e) => setPhone(e.target.value)} required />
          </div>

          <div className="input-group">
            <label htmlFor="message">Message</label>
            <textarea id="message" placeholder="Enter your message" onChange={(e) => setMessage(e.target.value)} rows="4"></textarea>
          </div>

          <button type="submit" className="submit-button">Submit</button>
        </form>
      </div>
    </div>
  );
}
