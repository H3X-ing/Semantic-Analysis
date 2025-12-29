# ml_loader.py
import os
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class MLLoader:
    """Handles loading and managing ML models separately"""
    def __init__(self, ml_assets_dir="gui_ml_assets"):
        self.ml_assets_dir = ml_assets_dir
        self.tfidf_vectorizer = None
        self.label_mapping = None
        self.models = {}
        self.loaded = False
        
        # Define model paths
        self.model_paths = {
            "SVM": os.path.join(ml_assets_dir, "svm.pkl"),
            "Random Forest": os.path.join(ml_assets_dir, "random_forest.pkl"),
            "Naive Bayes": os.path.join(ml_assets_dir, "naive_bayes.pkl"),
            "Logistic Regression": os.path.join(ml_assets_dir, "logistic_regression.pkl"),
            "TF-IDF Vectorizer": os.path.join(ml_assets_dir, "tfidf_vectorizer.pkl"),
            "Label Mapping": os.path.join(ml_assets_dir, "label_mapping.pkl")
        }
    
    def load_all(self):
        """Load all ML models and assets"""
        try:
            print("🔄 Loading ML models...")
            
            # Check if directory exists
            if not os.path.exists(self.ml_assets_dir):
                raise FileNotFoundError(f"ML assets directory not found: {self.ml_assets_dir}")
            
            # Load TF-IDF vectorizer
            tfidf_path = self.model_paths["TF-IDF Vectorizer"]
            if os.path.exists(tfidf_path):
                self.tfidf_vectorizer = joblib.load(tfidf_path)
                print(f"✅ TF-IDF Vectorizer loaded from {tfidf_path}")
            else:
                print(f"⚠️ TF-IDF Vectorizer not found at {tfidf_path}")
                return False
            
            # Load label mapping
            label_path = self.model_paths["Label Mapping"]
            if os.path.exists(label_path):
                self.label_mapping = joblib.load(label_path)
                print(f"✅ Label mapping loaded from {label_path}")
            else:
                print(f"⚠️ Label mapping not found at {label_path}")
            
            # Load ML models
            ml_models = ["SVM", "Random Forest", "Naive Bayes", "Logistic Regression"]
            for model_name in ml_models:
                model_path = self.model_paths[model_name]
                if os.path.exists(model_path):
                    try:
                        self.models[model_name] = joblib.load(model_path)
                        print(f"✅ {model_name} loaded from {model_path}")
                    except Exception as e:
                        print(f"❌ Failed to load {model_name}: {e}")
                        continue
                else:
                    print(f"⚠️ {model_name} not found at {model_path}")
            
            self.loaded = len(self.models) > 0
            return self.loaded
            
        except Exception as e:
            print(f"❌ Error loading ML models: {e}")
            return False
    
    def unload_all(self):
        """Unload all ML models"""
        self.tfidf_vectorizer = None
        self.label_mapping = None
        self.models = {}
        self.loaded = False
        print("✅ ML models unloaded")
    
    def predict(self, model_name, text):
        """Make prediction using a specific ML model"""
        if not self.loaded:
            raise ValueError("ML models not loaded")
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        if self.tfidf_vectorizer is None:
            raise ValueError("TF-IDF vectorizer not loaded")
        
        # Transform text
        text_tfidf = self.tfidf_vectorizer.transform([text])
        
        # Get model
        model = self.models[model_name]
        
        # Get prediction
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(text_tfidf)[0]
        else:
            # For models without predict_proba
            if hasattr(model, 'decision_function'):
                scores = model.decision_function(text_tfidf)
                if len(scores.shape) > 1:
                    scores = scores[0]
                # Convert scores to probabilities using softmax
                exp_scores = np.exp(scores - np.max(scores))
                probabilities = exp_scores / exp_scores.sum()
            else:
                # Fallback: equal probabilities
                probabilities = np.ones(4) / 4  # 4 categories
        
        predicted_idx = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_idx])
        
        # Get label
        categories = ["Normal", "Harassment", "Fraudulent", "Suspicious"]
        label = categories[predicted_idx]  # Default
        
        # Try to use label mapping if available
        if self.label_mapping is not None:
            try:
                prediction = model.predict(text_tfidf)[0]
                if prediction in self.label_mapping:
                    label = self.label_mapping[prediction]
                elif str(prediction) in self.label_mapping:
                    label = self.label_mapping[str(prediction)]
            except:
                pass
        
        return {
            'label': label,
            'confidence': confidence,
            'probabilities': probabilities,
            'success': True
        }
    
    def get_loaded_models(self):
        """Get list of loaded model names"""
        return list(self.models.keys())