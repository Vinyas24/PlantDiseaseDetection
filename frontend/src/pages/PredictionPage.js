import { useState, useRef } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Upload, ArrowLeft, Leaf, AlertCircle } from "lucide-react";
import axios from "axios";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const PredictionPage = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef(null);
  const navigate = useNavigate();

  const handleFileSelect = (file) => {
    if (!file) return;

    if (!file.type.startsWith('image/')) {
      setError('Please select a valid image file (JPG, PNG, etc.)');
      return;
    }

    if (file.size > 10 * 1024 * 1024) {
      setError('File size must be less than 10MB');
      return;
    }

    setSelectedFile(file);
    setError(null);
    setPrediction(null);


    const reader = new FileReader();
    reader.onloadend = () => {
      setPreviewUrl(reader.result);
    };
    reader.readAsDataURL(file);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setDragOver(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    handleFileSelect(file);
  };

  const handleFileInputChange = (e) => {
    const file = e.target.files[0];
    handleFileSelect(file);
  };

  const handlePredict = async () => {
    if (!selectedFile) {
      setError('Please select an image first');
      return;
    }

    setLoading(true);
    setError(null);
    setPrediction(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);

      const response = await axios.post(`${API}/predict`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setPrediction(response.data);
    } catch (err) {
      console.error('Prediction error:', err);
      if (err.response?.data?.detail) {
        setError(err.response.data.detail);
      } else if (err.response?.status === 400) {
        setError('Invalid file format. Please upload a valid leaf image.');
      } else if (err.response?.status >= 500) {
        setError('Server error. Please try again later.');
      } else {
        setError('Failed to predict disease. Please check your connection and try again.');
      }
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setPrediction(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const formatClassName = (className) => {
    return className.replace(/_/g, ' ').replace(/___/g, ' - ');
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.9) return '#166534';
    if (confidence >= 0.75) return '#15803d';
    return '#ca8a04';
  };

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
              <Link to="/" data-testid="nav-home-link">Home</Link>
            </li>
          </ul>
        </div>
      </nav>

      <div className="prediction-container" data-testid="prediction-container">
        <div className="prediction-card">
          <h2 data-testid="prediction-title">Plant Disease Detection</h2>
          <p style={{ textAlign: "center", color: "#15803d", marginBottom: "2rem" }}>
            Upload a clear photo of a plant leaf to detect diseases
          </p>

          {!previewUrl && (
            <div
              className={`upload-area ${dragOver ? 'drag-over' : ''}`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              data-testid="upload-area"
            >
              <div className="upload-icon">
                <Upload size={48} />
              </div>
              <p className="upload-text">Drag & drop your image here</p>
              <p className="upload-subtext">or click to browse</p>
              <p className="upload-subtext" style={{ marginTop: "0.5rem", fontSize: "0.75rem" }}>
                Supported formats: JPG, PNG, JPEG (Max 10MB)
              </p>
            </div>
          )}

          <input
            ref={fileInputRef}
            type="file"
            className="file-input"
            accept="image/*"
            onChange={handleFileInputChange}
            data-testid="file-input"
          />

          {previewUrl && (
            <div className="image-preview" data-testid="image-preview">
              <img src={previewUrl} alt="Selected leaf" />
              <button
                onClick={handleReset}
                style={{
                  marginTop: "1rem",
                  padding: "0.5rem 1.5rem",
                  background: "rgba(239, 68, 68, 0.1)",
                  color: "#dc2626",
                  border: "1px solid rgba(239, 68, 68, 0.3)",
                  borderRadius: "50px",
                  cursor: "pointer",
                  fontWeight: "500",
                }}
                data-testid="reset-button"
              >
                Change Image
              </button>
            </div>
          )}

          {selectedFile && !prediction && (
            <button
              className="predict-button"
              onClick={handlePredict}
              disabled={loading}
              data-testid="predict-button"
            >
              {loading ? 'Analyzing...' : 'Predict Disease'}
            </button>
          )}

          {/* Loading Spinner */}
          {loading && (
            <div className="loading-spinner" data-testid="loading-spinner">
              <div className="spinner"></div>
              <span>Analyzing leaf image...</span>
            </div>
          )}

          {/* Error Message */}
          {error && (
            <div className="error-box" data-testid="error-message">
              <AlertCircle size={24} style={{ marginBottom: "0.5rem" }} />
              <p>{error}</p>
            </div>
          )}

          {/* Prediction Results */}
          {prediction && (
            <div className="result-box" data-testid="result-box">
              <h3>Detection Results</h3>
              <div className="result-item">
                <span className="result-label">Predicted Disease:</span>
                <span className="result-value" data-testid="predicted-class">
                  {formatClassName(prediction.predicted_class)}
                </span>
              </div>
              <div className="result-item">
                <span className="result-label">Confidence:</span>
                <span 
                  className="result-value" 
                  style={{ 
                    color: getConfidenceColor(prediction.confidence),
                    fontWeight: "700",
                    fontSize: "1.1rem"
                  }}
                  data-testid="confidence-score"
                >
                  {(prediction.confidence * 100).toFixed(2)}%
                </span>
              </div>

              {/* Disease Description */}
              {prediction.description && (
                <div style={{ 
                  marginTop: "1.5rem", 
                  padding: "1rem", 
                  background: "rgba(255, 255, 255, 0.9)",
                  borderRadius: "12px",
                  borderLeft: "4px solid #22c55e"
                }}>
                  <h4 style={{ 
                    fontFamily: "'Space Grotesk', sans-serif",
                    color: "#166534",
                    fontSize: "1.1rem",
                    marginBottom: "0.5rem"
                  }}>
                    Description
                  </h4>
                  <p style={{ 
                    color: "#15803d",
                    lineHeight: "1.6",
                    fontSize: "0.95rem"
                  }} data-testid="disease-description">
                    {prediction.description}
                  </p>
                </div>
              )}

              {/* Possible Steps */}
              {prediction.possible_steps && (
                <div style={{ 
                  marginTop: "1rem", 
                  padding: "1rem", 
                  background: "rgba(255, 255, 255, 0.9)",
                  borderRadius: "12px",
                  borderLeft: "4px solid #16a34a"
                }}>
                  <h4 style={{ 
                    fontFamily: "'Space Grotesk', sans-serif",
                    color: "#166534",
                    fontSize: "1.1rem",
                    marginBottom: "0.5rem"
                  }}>
                    Recommended Actions
                  </h4>
                  <ul style={{ 
                    color: "#15803d",
                    lineHeight: "1.8",
                    fontSize: "0.95rem",
                    paddingLeft: "1.5rem",
                    margin: 0
                  }} data-testid="recommended-actions">
                    {prediction.possible_steps.split(',').map((step, index) => (
                      <li key={index} style={{ marginBottom: "0.5rem" }}>
                        {step.trim()}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              <button
                onClick={handleReset}
                className="predict-button"
                style={{ marginTop: "1.5rem" }}
                data-testid="analyze-another-button"
              >
                Analyze Another Image
              </button>
            </div>
          )}

          {/* Back Button */}
          <Link to="/" className="back-button" data-testid="back-to-home-button">
            <ArrowLeft size={20} />
            Back to Home
          </Link>
        </div>
      </div>
    </div>
  );
};

export default PredictionPage;