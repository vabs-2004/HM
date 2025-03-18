import React from "react";
import cybersvg from '../assets/police-logo.jpg';
import cyber2svg from '../assets/cyber2.jpg';
import cyber3svg from '../assets/cyber3.png';
import cyber4svg from '../assets/cyber4.jpg';
import './CyberSecurityLanding.css'


export default function CyberSecurityLanding() {
  return (
    <div className="container">
      <nav className="navbar">
        <h1 className="logo"><a className="logo" href="/">TruthShield</a></h1>
        <div className="nav-links">
          <a href="/" className="active">Home</a>
          <a href="/details">Contact us</a>
          <a href="/signup">Sign Up</a>
          <a href="/login">Login</a>
        </div>
      </nav>

      <div className="content large-content">
        <div className="text-section">
          <h2 className="title">Empowering You in the Digital Age</h2>
          <p className="description">Combat misinformation and discover the truth behind the headlines. Our tool helps you detect fake news and ensure you're always informed with reliable information.</p>
          <button className="purchase-button"><a href="/fakedetection">Detect Fake News</a></button>
        </div>
        
        <div className="image-section">
          <img src={cybersvg} alt="Cyber Security Shield" className="cyber-image" />
        </div>
      </div>
      
      <div className="about-section">
        <div className="about-images">
          <img src={cyber3svg} alt="Digital Security" className="about-image" />
          <img src={cyber2svg} alt="Hacker Keyboard" className="about-image overlay" />
        </div>
        <div className="about-text">
          <h3 className="about-title">Discover Our Journey Protecting Your Digital World With Expertise And Care</h3>
          <p className="about-description">With years of experience and dedication, we are committed to safeguarding your online presence. Our mission is to empower users by identifying and combating digital threats, ensuring that you stay informed and protected in this ever-evolving digital landscape.</p>
          <button className="read-more"><a className="link" href="/video">Detect Fake Video</a></button>
        </div>
      </div>
      
      <div className="services-section">
        <h3 className="services-title">Protecting Your Digital Assets Expertly</h3>
        <p className="services-description">Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vestibulum congue metus quis accumsan euismod. Maecenas sed est mollis.</p>
        <div className="services-container">
          <div className="service-card">
            <div className="service-icon">🔒</div>
            <h4 className="service-title">Cyber Security Assessment</h4>
            <p className="service-text">Our comprehensive services are designed to safeguard your digital world. We offer tailored solutions to protect your valuable assets, ensuring that your business stays secure in the face of evolving cyber threats.</p>
          </div>
          <div className="service-card">
            <div className="service-icon">🛡️</div>
            <h4 className="service-title">Intrusion Detection and Prevention</h4>
            <p className="service-text">Detect and prevent unauthorized access to your systems by leveraging advanced technologies to monitor and protect your network in real time.</p>
          </div>
          <div className="service-card">
            <div className="service-icon">🔄</div>
            <h4 className="service-title">Incident Response and Recovery</h4>
            <p className="service-text">Respond swiftly and recover from security incidents to minimize damage and restore operations as quickly as possible with our expert support.</p>
          </div>
        </div>
      </div>
      
      <div className="features-section">
        <div className="features-image">
          <img src={cyber4svg} alt="Cyber Security Expert" className="feature-img" />
        </div>
        <div className="features-content" id="services">
          <h3 className="features-title">Key Service Features Protecting You</h3>
          <p className="features-description">Stay informed and safeguard against misleading content with our cutting-edge fake news detection system.</p>
          <div className="features-container">
            <div className="feature-card">
              <div className="feature-icon">📜</div>
              <h4 className="feature-title">Customized Security Solutions</h4>
              <p className="feature-text">Leverages AI-driven analysis to identify and flag deceptive or misleading news.</p>
            </div>
            <div className="feature-card">
              <div className="feature-icon">🔍</div>
              <h4 className="feature-title">Real-Time Fact Verification</h4>
              <p className="feature-text">Instantly cross-checks sources to provide accurate and reliable information.</p>
            </div>
            <div className="feature-card">
              <div className="feature-icon">⏳</div>
              <h4 className="feature-title">24/7 Incident Response</h4>
              <p className="feature-text">Constantly scans news sources and alerts you to potential misinformation.</p>
            </div>
            <div className="feature-card">
              <div className="feature-icon">📚</div>
              <h4 className="feature-title">User Training Programs</h4>
              <p className="feature-text">Equips users with tools and insights to recognize and avoid fake news.</p>
            </div>
          </div>
        </div>
      </div>

      <div className="footer">
        <div className="footer-column">
          <h2 className="footer-logo">TruthShield</h2>
          <p>Your reliable defense against misinformation and fake news.
            
          </p>
          <div className="social-icons">
            <span>📷</span>
            <span>📘</span>
            <span>🐦</span>
            <span>▶️</span>
          </div>
        </div>
        <div className="footer-column">
          <h3>Quick Links</h3>
          <p>Our Service</p>
          <p>About Us</p>
          <p>Pricing</p>
          <p>Testimonial</p>
        </div>
        <div className="footer-column">
          <h3>Contact Us</h3>
          <p>📧 hello@website.com</p>
          <p>📍 838 Cantt Sialkot, ENG</p>
          <p>📞 +02 5421234560</p>
        </div>
        <div className="footer-column">
          <h3>Newsletter</h3>
          <input type="text" placeholder="Enter your email" className="newsletter-input" />
          <button className="subscribe-button">Subscribe</button>
        </div>
      </div>
    </div>
  );
}