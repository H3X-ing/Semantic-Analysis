# loader.py - FINAL FIX WITH to_empty()
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib
import os
import traceback
from safetensors.torch import load_file

class ModelLoader:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Transformer Loader initialized on device: {self.device}")

    def load_model(self, name, path):
        print(f"Loading {name} from {path}")
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model path not found: {path}")
        
        # List files for debugging
        files = os.listdir(path)
        print(f"Files in directory: {files}")
        
        try:
            # 1. Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                path,
                local_files_only=True,
                use_fast=True
            )
            
            # 2. Check which model file exists
            safetensors_path = os.path.join(path, "model.safetensors")
            pytorch_path = os.path.join(path, "pytorch_model.bin")
            
            if os.path.exists(safetensors_path):
                print(f"Found safetensors file: {safetensors_path}")
                model_file = safetensors_path
                use_safetensors = True
            elif os.path.exists(pytorch_path):
                print(f"Found pytorch bin file: {pytorch_path}")
                model_file = pytorch_path
                use_safetensors = False
            else:
                raise FileNotFoundError(f"No model file found in {path}")
            
            # 3. Load config first
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(path, local_files_only=True)
            
            # 4. Create empty model
            model = AutoModelForSequenceClassification.from_config(config)
            
            # 5. Load weights using to_empty() for meta tensors
            print(f"Using to_empty() to load meta tensor model to {self.device}")
            model = model.to_empty(device=self.device)
            
            # 6. Load state dict
            if use_safetensors:
                # Load from safetensors
                state_dict = load_file(model_file, device="cpu")
            else:
                # Load from pytorch bin
                state_dict = torch.load(model_file, map_location="cpu")
            
            # 7. Load state dict into model
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                print(f"⚠️ Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"⚠️ Unexpected keys: {unexpected_keys}")
            
            model.eval()
            print(f"✅ {name} loaded successfully using to_empty() method")
            
            # 8. Load label encoder if exists
            le = None
            le_path = os.path.join(path, "label_encoder.pkl")
            if os.path.exists(le_path):
                le = joblib.load(le_path)
                print(f"✅ Label encoder loaded for {name}")
            else:
                print(f"⚠️ No label encoder found for {name}")
            
            return model, tokenizer, le
            
        except Exception as e:
            print(f"❌ Error loading {name}: {str(e)}")
            print(traceback.format_exc())
            raise

    def predict(self, name, model, tokenizer, text, le=None):
        try:
            # Tokenize input
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Get probabilities
            probs = torch.softmax(outputs.logits, dim=-1)
            idx = torch.argmax(probs, dim=1).item()
            conf = probs[0][idx].item()
            
            # Get label
            if le:
                label = le.inverse_transform([idx])[0]
            else:
                # Map to categories
                categories = ["Normal", "Harassment", "Fraudulent", "Suspicious"]
                if 0 <= idx < len(categories):
                    label = categories[idx]
                else:
                    label = f"Class {idx}"
            
            return label, conf, probs.cpu().numpy()[0]
            
        except Exception as e:
            print(f"❌ Error predicting with {name}: {str(e)}")
            print(traceback.format_exc())
            raise