import React from 'react';
import Header from './components/Header';
import HeroSection from './components/HeroSection';
import Services from './components/Services';
import Testimonials from './components/Testimonials';
import Footer from './components/Footer';
import './App.css';
import CyberSecurityLanding from './components/CyberSecurityLanding';
import LoginPage from './components/LoginPage';
import DetailsForm from './components/DetailsForm';
import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";
import FakeNewsDetector from './components/FakeNewsDetector';
import Signup from './components/SignUpPage';
import VideoPage from './components/VideoPage';

function App() {
  return (
    <Router>

    <Routes>
      <Route path="/" element={<CyberSecurityLanding />} />
      <Route path="/login" element={<LoginPage />} />
      <Route path="/signup" element={<Signup />} />
      <Route path="/details" element={<DetailsForm />} />
      <Route path="/fakedetection" element={<div className='fake-news-container'><FakeNewsDetector /></div>} />
      <Route path='/video' element={<VideoPage/>}/>
    </Routes>
  </Router>
  );
}

export default App;