import { Link } from "react-router-dom";
import { Leaf, Brain, Database, TrendingUp, ArrowLeft } from "lucide-react";

const ModelPage = () => {
  return (
    <div>
      <nav className="navbar scrolled">
        <div className="navbar-container">
          <Link to="/" className="navbar-logo">
            <Leaf size={28} />
            PlantAI
          </Link>
          <ul className="navbar-links">
            <li>
              <Link to="/">Home</Link>
            </li>
            <li>
              <Link to="/predict" className="nav-cta">Try Detection</Link>
            </li>
          </ul>
        </div>
      </nav>

      {/* Introduction */}
      <section className="content-section page-hero" data-testid="introduction-section">
        <h2 className="section-title">About the Model</h2>
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

      {/* Methodology */}
      <section className="content-section" data-testid="methodology-section">
        <h2 className="section-title">Methodology</h2>
        <div className="section-content">
          <p>
            Our approach employs transfer learning with ResNet50, a powerful convolutional neural network 
            pre-trained on ImageNet. This methodology allows us to leverage features learned from millions 
            of images while training only the classification head for our specific task.
          </p>
        </div>
        
        <div className="cards-grid">
          <div className="glass-card feature-card" data-testid="methodology-card-dataset">
            <div className="card-icon"><Database size={28} /></div>
            <h3>Dataset Preparation</h3>
            <p>
              Images are collected and organized into 39 disease categories. Each image undergoes preprocessing 
              including resizing to 224×224 pixels, normalization, and data augmentation (rotation, flipping, 
              brightness adjustment) to improve model generalization.
            </p>
          </div>
          
          <div className="glass-card feature-card" data-testid="methodology-card-architecture">
            <div className="card-icon"><Brain size={28} /></div>
            <h3>Model Architecture</h3>
            <p>
              We use ResNet50 as a frozen feature extractor with 23 million non-trainable parameters. 
              A custom classifier head is added: Global Average Pooling → Dense(256, ReLU) → Dropout(0.5) → 
              Dense(39, Softmax), totaling ~534k trainable parameters.
            </p>
          </div>
          
          <div className="glass-card feature-card" data-testid="methodology-card-training">
            <div className="card-icon"><TrendingUp size={28} /></div>
            <h3>Training Process</h3>
            <p>
              The model is trained using categorical cross-entropy loss and Adam optimizer. We employ early 
              stopping, learning rate reduction, and cross-validation to prevent overfitting and ensure 
              robust performance across different plant species and disease types.
            </p>
          </div>
        </div>
      </section>

      {/* Dataset */}
      <section className="content-section" data-testid="dataset-section">
        <h2 className="section-title">Dataset & Classes</h2>
        <div className="glass-card info-panel">
          <p>
            Our model is trained to identify 39 different plant disease categories across multiple crop species. 
            The dataset includes both diseased and healthy plant samples, ensuring the model can distinguish 
            between various conditions and provide accurate diagnoses.
          </p>
          <div className="dataset-details">
            <div className="dataset-item">
              <strong>Covered Crop Species</strong>
              <p>Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, 
              Potato, Raspberry, Soybean, Squash, Strawberry, Tomato</p>
            </div>
            <div className="dataset-item">
              <strong>Disease Categories</strong>
              <p>Includes Apple Scab, Black Rot, Powdery Mildew, 
              Common Rust, Leaf Blight, Bacterial Spot, Early/Late Blight, Leaf Mold, Target Spot, Mosaic Virus, 
              and many more, along with healthy leaf classifications.</p>
            </div>
          </div>
        </div>
      </section>

      {/* Performance */}
      <section className="content-section" data-testid="performance-section">
        <h2 className="section-title">Model Performance</h2>
        <div className="cards-grid metrics-grid">
          <div className="glass-card metric-card">
            <h3>Accuracy</h3>
            <div className="metric-value">99%</div>
            <p>Overall classification accuracy on test set</p>
          </div>
          <div className="glass-card metric-card">
            <h3>Precision</h3>
            <div className="metric-value">98%</div>
            <p>Weighted average precision across all classes</p>
          </div>
          <div className="glass-card metric-card">
            <h3>F1-Score</h3>
            <div className="metric-value">98%</div>
            <p>Harmonic mean of precision and recall</p>
          </div>
        </div>

        <div className="glass-card graph-container" data-testid="training-graph">
          <h3>Training & Validation Curves</h3>
          <img 
            src="https://customer-assets.emergentagent.com/job_agri-detect-3/artifacts/7g36o1pb_image.png" 
            alt="Training and Validation Accuracy/Loss Curves" 
          />
        </div>

        <div className="glass-card info-panel" style={{ marginTop: "2rem" }}>
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

      {/* Back link */}
      <section className="content-section" style={{ textAlign: "center", paddingTop: 0 }}>
        <Link to="/" className="back-button">
          <ArrowLeft size={20} />
          Back to Home
        </Link>
      </section>

      <footer className="footer">
        <p>© 2025 PlantAI — Plant Disease Detection System</p>
        <p style={{ marginTop: "0.5rem", fontSize: "0.8rem", opacity: 0.7 }}>
          Powered by ResNet50 · 39 Disease Categories · 99% Accuracy
        </p>
      </footer>
    </div>
  );
};

export default ModelPage;
