import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { Leaf, Brain, Database, TrendingUp, Sprout, Shield } from "lucide-react";

const HomePage = () => {
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 50);
    };
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const scrollToSection = (id) => {
    const element = document.getElementById(id);
    if (element) {
      element.scrollIntoView({ behavior: "smooth" });
    }
  };

  return (
    <div>
      {/* Navigation */}
      <nav className={`navbar ${scrolled ? "scrolled" : ""}`}>
        <div className="navbar-container">
          <Link to="/" className="navbar-logo">
            <Leaf size={28} />
            PlantAI
          </Link>
          <ul className="navbar-links">
            <li>
              <a href="#introduction" onClick={(e) => { e.preventDefault(); scrollToSection("introduction"); }}>
                Introduction
              </a>
            </li>
            <li>
              <a href="#methodology" onClick={(e) => { e.preventDefault(); scrollToSection("methodology"); }}>
                Methodology
              </a>
            </li>
            <li>
              <a href="#performance" onClick={(e) => { e.preventDefault(); scrollToSection("performance"); }}>
                Performance
              </a>
            </li>
            <li>
              <a href="#applications" onClick={(e) => { e.preventDefault(); scrollToSection("applications"); }}>
                Applications
              </a>
            </li>
            <li>
              <Link to="/predict" className="nav-cta" data-testid="nav-predict-button">
                Try Detection
              </Link>
            </li>
          </ul>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="hero-section" data-testid="hero-section">
        <div className="floating-shape shape-1"></div>
        <div className="floating-shape shape-2"></div>
        <div className="floating-shape shape-3"></div>
        
        <div className="hero-content">
          <div className="hero-text">
            <h1 data-testid="hero-title">Plant Disease Detection Using CNN</h1>
            <p data-testid="hero-subtitle">
              Leveraging transfer learning and ResNet50 architecture to accurately classify 39 plant diseases, 
              helping farmers protect their crops and ensure food security through AI-powered agricultural technology.
            </p>
            <Link to="/predict" className="hero-cta" data-testid="hero-cta-button">
              <Sprout size={20} />
              Start Detection
            </Link>
          </div>
          <div className="hero-image">
            <img 
              src="/green-leaves-background.webp" 
              alt="Plant disease detection" 
              style={{ width: "1000px", height: "500px", objectFit: "cover" }}
            />
          </div>
        </div>
      </section>

      <section id="introduction" className="content-section" data-testid="introduction-section">
        <h2 className="section-title">Introduction</h2>
        <div className="section-content">
          <p>
            Plant diseases pose a significant threat to agricultural productivity and food security worldwide. 
            Traditional disease detection methods rely on manual inspection by experts, which is time-consuming, 
            expensive, and often unavailable in remote areas. Our deep learning-based system addresses these 
            challenges by providing rapid, accurate, and accessible disease detection through simple leaf image analysis.
          </p>
          <p style={{ marginTop: "1rem" }}>
            This project utilizes state-of-the-art computer vision techniques and transfer learning to identify 
            plant diseases early, enabling timely intervention and treatment. By democratizing access to plant 
            disease diagnosis, we aim to support farmers in making informed decisions and protecting their crops.
          </p>
        </div>
      </section>

      {/* Methodology Section */}
      <section id="methodology" className="content-section" data-testid="methodology-section">
        <h2 className="section-title">Methodology</h2>
        <div className="section-content">
          <p>
            Our approach employs transfer learning with ResNet50, a powerful convolutional neural network 
            pre-trained on ImageNet. This methodology allows us to leverage features learned from millions 
            of images while training only the classification head for our specific task.
          </p>
        </div>
        
        <div className="cards-grid">
          <div className="info-card" data-testid="methodology-card-dataset">
            <h3><Database size={24} /> Dataset Preparation</h3>
            <p>
              Images are collected and organized into 39 disease categories. Each image undergoes preprocessing 
              including resizing to 224×224 pixels, normalization, and data augmentation (rotation, flipping, 
              brightness adjustment) to improve model generalization.
            </p>
          </div>
          
          <div className="info-card" data-testid="methodology-card-architecture">
            <h3><Brain size={24} /> Model Architecture</h3>
            <p>
              We use ResNet50 as a frozen feature extractor with 23 million non-trainable parameters. 
              A custom classifier head is added: Global Average Pooling → Dense(256, ReLU) → Dropout(0.5) → 
              Dense(39, Softmax), totaling ~534k trainable parameters.
            </p>
          </div>
          
          <div className="info-card" data-testid="methodology-card-training">
            <h3><TrendingUp size={24} /> Training Process</h3>
            <p>
              The model is trained using categorical cross-entropy loss and Adam optimizer. We employ early 
              stopping, learning rate reduction, and cross-validation to prevent overfitting and ensure 
              robust performance across different plant species and disease types.
            </p>
          </div>
        </div>
      </section>

      <section id="dataset" className="content-section" data-testid="dataset-section">
        <h2 className="section-title">Dataset & Classes</h2>
        <div className="section-content">
          <p>
            Our model is trained to identify 39 different plant disease categories across multiple crop species. 
            The dataset includes both diseased and healthy plant samples, ensuring the model can distinguish 
            between various conditions and provide accurate diagnoses.
          </p>
          <p style={{ marginTop: "1rem" }}>
            <strong>Covered Crop Species:</strong> Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, 
            Potato, Raspberry, Soybean, Squash, Strawberry, Tomato
          </p>
          <p style={{ marginTop: "1rem" }}>
            <strong>Disease Categories:</strong> Includes common diseases like Apple Scab, Black Rot, Powdery Mildew, 
            Common Rust, Leaf Blight, Bacterial Spot, Early/Late Blight, Leaf Mold, Target Spot, Mosaic Virus, 
            and many more, along with healthy leaf classifications.
          </p>
        </div>
      </section>

      <section id="performance" className="content-section" data-testid="performance-section">
        <h2 className="section-title">Model Performance</h2>
        <div className="cards-grid">
          <div className="info-card">
            <h3>Accuracy</h3>
            <p style={{ fontSize: "2.5rem", fontWeight: "700", color: "#166534", marginTop: "1rem" }}>
              99%
            </p>
            <p>Overall classification accuracy on test set</p>
          </div>
          <div className="info-card">
            <h3>Precision</h3>
            <p style={{ fontSize: "2.5rem", fontWeight: "700", color: "#166534", marginTop: "1rem" }}>
              98%
            </p>
            <p>Weighted average precision across all classes</p>
          </div>
          <div className="info-card">
            <h3>F1-Score</h3>
            <p style={{ fontSize: "2.5rem", fontWeight: "700", color: "#166534", marginTop: "1rem" }}>
              98%
            </p>
            <p>Harmonic mean of precision and recall</p>
          </div>
        </div>

        <div className="graph-container" data-testid="training-graph">
          <h3>Training & Validation Curves</h3>
          <img 
            src="https://customer-assets.emergentagent.com/job_agri-detect-3/artifacts/7g36o1pb_image.png" 
            alt="Training and Validation Accuracy/Loss Curves" 
          />
        </div>

        <div className="section-content" style={{ marginTop: "3rem" }}>
          <p>
            The model demonstrates excellent performance across all disease categories with minimal confusion 
            between similar-looking conditions. The classification report shows that 98% accuracy is achieved 
            across 9,219 test samples. Most classes achieve F1-scores above 0.95, with perfect scores (1.00) 
            for several categories including Blueberry healthy, Grape healthy, Orange Haunglongbing, and others. 
            The lowest F1-score (0.88) is observed for Tomato Early blight, indicating potential areas for 
            improvement in future model iterations.
          </p>
        </div>
      </section>

      <section id="applications" className="content-section" data-testid="applications-section">
        <h2 className="section-title">Real-World Applications</h2>
        <div className="cards-grid">
          <div className="info-card">
            <h3><Shield size={24} /> Crop Protection</h3>
            <p>
              Early disease detection enables timely intervention, preventing disease spread and minimizing 
              crop losses. Farmers can take targeted action before diseases reach critical stages.
            </p>
          </div>
          <div className="info-card">
            <h3><Sprout size={24} /> Sustainable Farming</h3>
            <p>
              Precise disease identification reduces unnecessary pesticide use, promoting environmentally 
              friendly farming practices and reducing chemical exposure to soil and water systems.
            </p>
          </div>
          <div className="info-card">
            <h3><TrendingUp size={24} /> Yield Optimization</h3>
            <p>
              By maintaining plant health through early disease management, farmers can maximize crop yields 
              and improve produce quality, leading to better market prices and profitability.
            </p>
          </div>
          <div className="info-card">
            <h3><Database size={24} /> Knowledge Sharing</h3>
            <p>
              The system can be deployed as a mobile app, providing instant expert-level diagnosis to farmers 
              in remote areas who lack access to agricultural specialists.
            </p>
          </div>
        </div>
      </section>

      <section id="about" className="content-section" data-testid="about-section">
        <h2 className="section-title">About This Project</h2>
        <div className="section-content">
          <p>
            This plant disease detection system was developed to address the critical need for accessible, 
            accurate, and rapid disease diagnosis in agriculture. By combining deep learning expertise with 
            agricultural domain knowledge, we've created a tool that can help protect global food security.
          </p>
          <p style={{ marginTop: "1rem" }}>
            The project demonstrates the power of transfer learning in agricultural applications, showing that 
            state-of-the-art AI technology can be effectively adapted to solve real-world problems with limited 
            computational resources and training data.
          </p>
          <div style={{ textAlign: "center", marginTop: "3rem" }}>
            <Link to="/predict" className="hero-cta" data-testid="about-cta-button">
              <Leaf size={20} />
              Try Disease Detection
            </Link>
          </div>
        </div>
      </section>

      <footer className="footer">
        <p>© 2025 Plant Disease Detection System | Built with Deep Learning & Transfer Learning</p>
        <p style={{ marginTop: "0.5rem", fontSize: "0.875rem" }}>
          Powered by ResNet50 Architecture | 39 Disease Categories | 96.8% Accuracy
        </p>
      </footer>
    </div>
  );
};

export default HomePage;