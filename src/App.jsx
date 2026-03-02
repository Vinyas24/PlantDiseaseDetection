import { BrowserRouter, Routes, Route } from "react-router-dom";
import HomePage from "./pages/HomePage.jsx";
import ModelPage from "./pages/ModelPage.jsx";
import PredictionPage from "./pages/PredictionPage.jsx";
import "./App.css";

function App() {
  return (
    <div className="App">
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/model" element={<ModelPage />} />
          <Route path="/predict" element={<PredictionPage />} />
        </Routes>
      </BrowserRouter>
    </div>
  );
}

export default App;
