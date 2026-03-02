import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { Leaf, Sprout, Brain, Shield, TrendingUp, ArrowRight } from "lucide-react";

const HomePage = () => {
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 50);
    };
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  return (
    <div>
      <nav className={`navbar ${scrolled ? "scrolled" : ""}`}>
        <div className="navbar-container">
          <Link to="/" className="navbar-logo">
            <Leaf size={28} />
            PlantAI
          </Link>
          <ul className="navbar-links">
            <li>
              <Link to="/model">Model Details</Link>
            </li>
            <li>
              <Link to="/predict" className="nav-cta" data-testid="nav-predict-button">
                Try Detection
              </Link>
            </li>
          </ul>
        </div>
      </nav>

      {/* Hero */}
      <section className="hero-section" data-testid="hero-section">
        <div className="hero-glow glow-1"></div>
        <div className="hero-glow glow-2"></div>
        <div className="hero-glow glow-3"></div>

        <div className="hero-content">
          <div className="hero-text">
            <div className="hero-badge">
              <Sprout size={16} />
              AI-Powered Agriculture
            </div>
            <h1 data-testid="hero-title">
              Plant Disease Detection<br />
              <span className="gradient-text">Using Deep Learning</span>
            </h1>
            <p data-testid="hero-subtitle">
              Leveraging transfer learning and ResNet50 architecture to accurately classify 
              39 plant diseases, helping farmers protect their crops through AI-powered technology.
            </p>
            <div className="hero-actions">
              <Link to="/predict" className="hero-cta" data-testid="hero-cta-button">
                <Sprout size={20} />
                Start Detection
                <span className="cta-arrow">→</span>
              </Link>
              <Link to="/model" className="hero-secondary">
                <Brain size={18} />
                View Model Details
              </Link>
            </div>
          </div>

          <div className="hero-visual">
            <div className="glass-card hero-stats-card">
              <div className="hero-card-icon">
                <Leaf size={40} />
              </div>
              <div className="hero-card-stats">
                <div className="stat-item">
                  <span className="stat-number">39</span>
                  <span className="stat-label">Disease Classes</span>
                </div>
                <div className="stat-item">
                  <span className="stat-number">99%</span>
                  <span className="stat-label">Accuracy</span>
                </div>
                <div className="stat-item">
                  <span className="stat-number">14</span>
                  <span className="stat-label">Crop Species</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Features */}
      <section className="content-section">
        <h2 className="section-title">Why PlantAI?</h2>
        <div className="cards-grid">
          <div className="glass-card feature-card">
            <div className="card-icon"><Shield size={28} /></div>
            <h3>Crop Protection</h3>
            <p>
              Early disease detection enables timely intervention, preventing disease spread 
              and minimizing crop losses before they reach critical stages.
            </p>
          </div>
          <div className="glass-card feature-card">
            <div className="card-icon"><Sprout size={28} /></div>
            <h3>Sustainable Farming</h3>
            <p>
              Precise disease identification reduces unnecessary pesticide use, promoting 
              environmentally friendly farming practices.
            </p>
          </div>
          <div className="glass-card feature-card">
            <div className="card-icon"><TrendingUp size={28} /></div>
            <h3>Yield Optimization</h3>
            <p>
              Maintain plant health through early disease management, maximizing crop yields 
              and improving produce quality.
            </p>
          </div>
          <div className="glass-card feature-card">
            <div className="card-icon"><Brain size={28} /></div>
            <h3>AI-Powered</h3>
            <p>
              Built on ResNet50 with transfer learning, providing expert-level diagnosis 
              accessible to farmers worldwide.
            </p>
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="content-section cta-section">
        <div className="glass-card cta-card">
          <h2>Ready to detect plant diseases?</h2>
          <p>Upload a leaf image and get instant AI-powered diagnosis with treatment recommendations.</p>
          <div className="cta-actions">
            <Link to="/predict" className="hero-cta">
              <Leaf size={20} />
              Try Disease Detection
            </Link>
            <Link to="/model" className="hero-secondary">
              Learn about the model
              <ArrowRight size={16} />
            </Link>
          </div>
        </div>
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

export default HomePage;
