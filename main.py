# main.py
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import os
import traceback
import datetime
import email
from email import policy
from email.parser import BytesParser
from bs4 import BeautifulSoup
import re
import numpy as np
import pandas as pd
import joblib

# Import the loaders
from ml_loader import MLLoader
from loader import ModelLoader as TransformerLoader  # Your transformer loader
from rnn_loader import RNNModelLoader  # Your RNN loader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ML Models Paths
ML_MODELS_DIR = os.path.join(BASE_DIR, "gui_ml_assets")

# Deep Learning Models Paths
DL_MODEL_PATHS = {
    "DeBERTa v3": os.path.join(BASE_DIR, "models", "DeBERTa_email_model"),
    "RoBERTa": os.path.join(BASE_DIR, "models", "RoBERTa_email_model"),
    "LSTM-GRU": os.path.join(BASE_DIR, "models", "LSTM_GRU_email_model")
}

EMAIL_CATEGORIES = ["Normal", "Harassment", "Fraudulent", "Suspicious"]

class EmailParser:
    def extract_body(self, path):
        """
        Extract text body from email file (.eml)
        Handles both plain text and HTML emails
        """
        try:
            with open(path, "rb") as f:
                msg = BytesParser(policy=policy.default).parse(f)
            
            body_parts = []
            
            # Add email headers for context
            body_parts.append(f"Subject: {msg.get('subject', 'No Subject')}")
            body_parts.append(f"From: {msg.get('from', 'Unknown Sender')}")
            body_parts.append(f"Date: {msg.get('date', 'Unknown Date')}")
            body_parts.append("")  # Empty line
            
            # Extract main body content
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_maintype() == 'text':
                        body_content = self._extract_part_content(part)
                        if body_content:
                            body_parts.append(body_content)
            else:
                body_content = self._extract_part_content(msg)
                if body_content:
                    body_parts.append(body_content)
            
            # Combine all parts
            full_text = "\n".join(body_parts)
            
            # Clean up text
            cleaned_text = self._clean_text(full_text)
            
            return cleaned_text
            
        except Exception as e:
            raise Exception(f"Failed to parse email: {str(e)}")
    
    def _extract_part_content(self, part):
        """Extract content from a single email part"""
        try:
            # Get content type
            content_type = part.get_content_type()
            
            # Get payload
            payload = part.get_payload(decode=True)
            if not payload:
                return ""
            
            # Try to decode with correct charset
            charset = part.get_content_charset() or 'utf-8'
            
            # Handle different encodings
            try:
                text = payload.decode(charset, errors='replace')
            except (LookupError, UnicodeDecodeError):
                # Fallback to utf-8
                text = payload.decode('utf-8', errors='replace')
            
            # If HTML, extract text
            if content_type == 'text/html':
                text = self._html_to_text(text)
            
            return text.strip()
            
        except Exception as e:
            # If decoding fails, return empty string
            return ""
    
    def _html_to_text(self, html_content):
        """Convert HTML to plain text"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text with proper spacing
            text = soup.get_text(separator=' ')
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
        except:
            # If BeautifulSoup fails, return original with HTML tags removed
            return re.sub(r'<[^>]+>', ' ', html_content)
    
    def _clean_text(self, text):
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common email signatures patterns
        signature_patterns = [
            r'--+\s*$',
            r'^Sent from my .+$',
            r'^Best regards,.+$',
            r'^Thanks,.+$',
            r'^Regards,.+$',
            r'^Sincerely,.+$',
        ]
        
        for pattern in signature_patterns:
            text = re.sub(pattern, '', text, flags=re.MULTILINE | re.IGNORECASE)
        
        # Remove empty lines
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        text = '\n'.join(lines)
        
        return text.strip()

class ModelLoader:
    """Handles loading and managing DL models only"""
    def __init__(self):
        self.loaded_models = {}
        # Initialize specialized loaders
        self.transformer_loader = TransformerLoader()  # For DeBERTa and RoBERTa
        self.rnn_loader = RNNModelLoader()  # For LSTM-GRU
        
    def load_all_models(self, callback=None):
        """Load all DL models in parallel"""
        threads = []
        for model_name in DL_MODEL_PATHS.keys():
            thread = threading.Thread(target=self._load_model_thread, args=(model_name, callback))
            threads.append(thread)
            thread.start()
        
        return threads
    
    def _load_model_thread(self, model_name, callback):
        """Thread function to load a single DL model"""
        try:
            path = DL_MODEL_PATHS.get(model_name)
            if not path or not os.path.exists(path):
                if callback:
                    callback(model_name, False, f"Model directory not found: {path}")
                return
            
            model_data = {}
            
            if model_name == "LSTM-GRU":
                # Load RNN model using your rnn_loader
                try:
                    model, tokenizer, label_encoder = self.rnn_loader.load_model(path)
                    
                    model_data['model'] = model
                    model_data['tokenizer'] = tokenizer
                    model_data['label_encoder'] = label_encoder
                    model_data['type'] = 'lstm_gru'
                    model_data['name'] = model_name
                    model_data['loader'] = self.rnn_loader  # Store loader reference
                    
                    self.loaded_models[model_name] = model_data
                    
                    if callback:
                        callback(model_name, True, None)
                        
                except Exception as e:
                    if callback:
                        callback(model_name, False, f"RNN error: {str(e)}")
                
            else:
                # Load Transformer model using your loader.py
                try:
                    # Use your transformer loader with proper error handling
                    model, tokenizer, le = self.transformer_loader.load_model(model_name, path)
                    
                    model_data['model'] = model
                    model_data['tokenizer'] = tokenizer
                    model_data['label_encoder'] = le
                    model_data['type'] = 'transformer'
                    model_data['name'] = model_name
                    model_data['device'] = self.transformer_loader.device
                    model_data['loader'] = self.transformer_loader  # Store loader reference
                    
                    self.loaded_models[model_name] = model_data
                    
                    if callback:
                        callback(model_name, True, None)
                        
                except Exception as e:
                    error_msg = f"Transformers error: {str(e)}\n"
                    error_msg += f"Model path: {path}\n"
                    if os.path.exists(path):
                        files = os.listdir(path)
                        error_msg += f"Files in directory: {files}\n"
                    
                    print(error_msg)
                    if callback:
                        callback(model_name, False, f"Transformers error: {str(e)}")
                
        except Exception as e:
            if callback:
                callback(model_name, False, str(e))

class EmailClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("📧 AI SEMANTIC ANALYZER (DL + ML)")
        self.root.state("zoomed")
        
        # Initialize parsers and loaders
        self.parser = EmailParser()
        self.dl_loader = ModelLoader()  # DL models only
        self.ml_loader = MLLoader(ML_MODELS_DIR)  # ML models only
        
        # Model state variables
        self.current_file_path = None
        self.email_text = ""
        
        # Results storage for all models
        self.all_results = {}
        
        self.create_gui()

    def create_gui(self):
        # Configure grid
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Main container
        canvas = tk.Canvas(self.root)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        main_frame = ttk.Frame(canvas, padding="10")
        canvas.create_window((0, 0), window=main_frame, anchor="nw")

        def _on_frame_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        main_frame.bind("<Configure>", _on_frame_configure)
        main_frame.columnconfigure(0, weight=1)

        # Mouse wheel scrolling
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(-1*(e.delta // 120), "units"))
        
        # ===== HEADER =====
        header_frame = ttk.Frame(main_frame)
        header_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        
        ttk.Label(header_frame,
                 text="📧 AI SEMANTIC ANALYZER (DL + ML)",
                 font=("Arial", 16, "bold")).pack()
        
        ttk.Label(header_frame,
                 text="Classify emails into: Normal, Harassment, Fraudulent, or Suspicious",
                 font=("Arial", 10)).pack()
        
        # ===== MODEL LOADING SECTION =====
        model_frame = ttk.LabelFrame(main_frame, text="1. Load Models", padding="10")
        model_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        
        model_btn_frame = ttk.Frame(model_frame)
        model_btn_frame.pack(fill="x", pady=(0, 10))
        
        ttk.Button(model_btn_frame, text="Load DL Models", 
                  command=self.load_dl_models,
                  style="Accent.TButton").pack(side="left", padx=(0, 10))
        
        ttk.Button(model_btn_frame, text="Load ML Models",
                  command=self.load_ml_models).pack(side="left", padx=(0, 10))
        
        ttk.Button(model_btn_frame, text="Load All Models",
                  command=self.load_all_models).pack(side="left", padx=(0, 10))
        
        ttk.Button(model_btn_frame, text="Unload All",
                  command=self.unload_all_models).pack(side="left")
        
        # Model status labels
        self.model_status_frame = ttk.Frame(model_frame)
        self.model_status_frame.pack(fill="x")
        
        # Create frames for DL and ML models
        dl_frame = ttk.Frame(self.model_status_frame)
        dl_frame.pack(fill="x", pady=(0, 10))
        
        ml_frame = ttk.Frame(self.model_status_frame)
        ml_frame.pack(fill="x")
        
        ttk.Label(dl_frame, text="Deep Learning Models:", font=("Arial", 9, "bold")).pack(anchor="w")
        dl_subframe = ttk.Frame(dl_frame)
        dl_subframe.pack(fill="x", pady=(5, 0))
        
        ttk.Label(ml_frame, text="Machine Learning Models:", font=("Arial", 9, "bold")).pack(anchor="w")
        ml_subframe = ttk.Frame(ml_frame)
        ml_subframe.pack(fill="x", pady=(5, 0))
        
        self.model_status_labels = {}
        
        # DL models status
        for i, model_name in enumerate(DL_MODEL_PATHS.keys()):
            frame = ttk.Frame(dl_subframe)
            frame.pack(side="left", padx=(0, 20))
            
            label = ttk.Label(frame, text=model_name, width=15, anchor="w")
            label.pack(side="left", padx=(0, 10))
            
            status = ttk.Label(frame, text="❌ Not loaded", foreground="red")
            status.pack(side="left")
            
            self.model_status_labels[model_name] = status
        
        # ML models status
        ml_models_list = ["SVM", "Random Forest", "Naive Bayes", "Logistic Regression"]
        for i, model_name in enumerate(ml_models_list):
            frame = ttk.Frame(ml_subframe)
            frame.pack(side="left", padx=(0, 20))
            
            label = ttk.Label(frame, text=model_name, width=15, anchor="w")
            label.pack(side="left", padx=(0, 10))
            
            status = ttk.Label(frame, text="❌ Not loaded", foreground="red")
            status.pack(side="left")
            
            self.model_status_labels[model_name] = status
        
        # ===== FILE SELECTION =====
        file_frame = ttk.LabelFrame(main_frame, text="2. Select Email File", padding="10")
        file_frame.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        
        self.file_var = tk.StringVar(value="No file selected")
        
        file_top_frame = ttk.Frame(file_frame)
        file_top_frame.pack(fill="x", pady=(0, 10))
        
        ttk.Label(file_top_frame, text="File:", font=("Arial", 10, "bold")).pack(
            side="left", padx=(0, 10))
        
        ttk.Label(file_top_frame, textvariable=self.file_var, foreground="blue").pack(
            side="left", fill="x", expand=True)
        
        file_btn_frame = ttk.Frame(file_frame)
        file_btn_frame.pack()
        
        ttk.Button(file_btn_frame, text="Browse File", command=self.browse_file,
                  width=15).pack(side="left", padx=5)
        ttk.Button(file_btn_frame, text="Parse Email", command=self.parse_email,
                  width=15).pack(side="left", padx=5)
        
        # ===== EMAIL CONTENT =====
        content_frame = ttk.LabelFrame(main_frame, text="3. Email Content", padding="10")
        content_frame.grid(row=3, column=0, sticky="nsew", pady=(0, 10))
        content_frame.rowconfigure(0, weight=1)
        content_frame.columnconfigure(0, weight=1)
        
        # Text area with scrollbar
        text_frame = ttk.Frame(content_frame)
        text_frame.grid(row=0, column=0, sticky="nsew")
        text_frame.rowconfigure(0, weight=1)
        text_frame.columnconfigure(0, weight=1)
        
        self.text_area = tk.Text(text_frame, wrap="word", height=10,
                                font=("Courier New", 10))
        self.text_area.grid(row=0, column=0, sticky="nsew")
        
        scrollbar = ttk.Scrollbar(text_frame, command=self.text_area.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.text_area.config(yscrollcommand=scrollbar.set)
        
        self.text_info = ttk.Label(content_frame, text="No email loaded",
                                  foreground="gray")
        self.text_info.grid(row=1, column=0, sticky="w", pady=(5, 0))
        
        # ===== ACTION BUTTONS =====
        action_frame = ttk.Frame(main_frame)
        action_frame.grid(row=4, column=0, pady=(0, 10))
        
        ttk.Button(action_frame, text="🎯 Classify with All Models", 
                  command=self._classify_all_thread,
                  width=25, style="Accent.TButton").pack(side="left", padx=5)
        ttk.Button(action_frame, text="🔬 Classify with ML Models",
                  command=self._classify_ml_only,
                  width=25).pack(side="left", padx=5)
        ttk.Button(action_frame, text="🧠 Classify with DL Models",
                  command=self._classify_dl_only,
                  width=25).pack(side="left", padx=5)
        ttk.Button(action_frame, text="🔄 Reset", command=self.reset,
                  width=15).pack(side="left", padx=5)
        ttk.Button(action_frame, text="🗑️ Clear", command=self.clear_text,
                  width=15).pack(side="left", padx=5)
        ttk.Button(action_frame, text="💾 Save", command=self.save_results,
                  width=15).pack(side="left", padx=5)
        
        # ===== COMPARISON RESULTS =====
        result_frame = ttk.LabelFrame(main_frame, text="4. Model Comparison Results", padding="10")
        result_frame.grid(row=5, column=0, sticky="nsew", pady=(0, 10))
        result_frame.columnconfigure(0, weight=1)
        
        # Create results table
        table_frame = ttk.Frame(result_frame)
        table_frame.grid(row=0, column=0, sticky="nsew", pady=(5, 0))
        
        # Table header
        headers = ["Model", "Type", "Prediction", "Confidence"] + EMAIL_CATEGORIES
        for col, header in enumerate(headers):
            if col == 0:
                width = 15
            elif col == 1:
                width = 10
            elif col == 2:
                width = 15
            elif col == 3:
                width = 12
            else:
                width = 10
            
            label = ttk.Label(table_frame, text=header, font=("Arial", 9, "bold"),
                             relief="ridge", width=width, anchor="center")
            label.grid(row=0, column=col, sticky="nsew", padx=1, pady=1)
        
        # Table rows for all models
        self.result_rows = {}
        all_models = list(DL_MODEL_PATHS.keys()) + ["SVM", "Random Forest", "Naive Bayes", "Logistic Regression"]
        
        for row, model_name in enumerate(all_models, 1):
            row_widgets = {}
            
            # Model name
            label = ttk.Label(table_frame, text=model_name, relief="sunken",
                             width=15, anchor="w")
            label.grid(row=row, column=0, sticky="nsew", padx=1, pady=1)
            row_widgets['model'] = label
            
            # Model type
            model_type = "DL" if model_name in DL_MODEL_PATHS else "ML"
            label = ttk.Label(table_frame, text=model_type, relief="sunken",
                             width=10, anchor="center")
            label.grid(row=row, column=1, sticky="nsew", padx=1, pady=1)
            row_widgets['type'] = label
            
            # Prediction
            label = ttk.Label(table_frame, text="Not run", relief="sunken",
                             width=15, anchor="center", foreground="gray")
            label.grid(row=row, column=2, sticky="nsew", padx=1, pady=1)
            row_widgets['prediction'] = label
            
            # Confidence
            label = ttk.Label(table_frame, text="0%", relief="sunken",
                             width=12, anchor="center")
            label.grid(row=row, column=3, sticky="nsew", padx=1, pady=1)
            row_widgets['confidence'] = label
            
            # Category probabilities
            for col, category in enumerate(EMAIL_CATEGORIES, 4):
                label = ttk.Label(table_frame, text="0%", relief="sunken",
                                 width=10, anchor="center")
                label.grid(row=row, column=col, sticky="nsew", padx=1, pady=1)
                row_widgets[category] = label
            
            self.result_rows[model_name] = row_widgets
        
        # Summary row
        summary_row = len(all_models) + 1
        ttk.Label(table_frame, text="Consensus:", font=("Arial", 9, "bold"),
                 relief="ridge", width=15, anchor="e").grid(
                 row=summary_row, column=0, sticky="nsew", padx=1, pady=1)
        
        ttk.Label(table_frame, text="All", font=("Arial", 9, "bold"),
                 relief="ridge", width=10, anchor="center").grid(
                 row=summary_row, column=1, sticky="nsew", padx=1, pady=1)
        
        self.consensus_pred = ttk.Label(table_frame, text="N/A", relief="sunken",
                                       width=15, anchor="center", font=("Arial", 9, "bold"))
        self.consensus_pred.grid(row=summary_row, column=2, sticky="nsew", padx=1, pady=1)
        
        self.consensus_conf = ttk.Label(table_frame, text="N/A", relief="sunken",
                                       width=12, anchor="center", font=("Arial", 9, "bold"))
        self.consensus_conf.grid(row=summary_row, column=3, sticky="nsew", padx=1, pady=1)
        
        for col, category in enumerate(EMAIL_CATEGORIES, 4):
            label = ttk.Label(table_frame, text="N/A", relief="sunken",
                             width=10, anchor="center", font=("Arial", 9, "bold"))
            label.grid(row=summary_row, column=col, sticky="nsew", padx=1, pady=1)
            setattr(self, f"consensus_{category.lower()}", label)
        
        # ===== STATUS BAR =====
        self.status_bar = ttk.Label(main_frame, text="Ready",
                                   relief="sunken", anchor="w")
        self.status_bar.grid(row=6, column=0, sticky="ew", pady=(10, 0))
        
        # Configure styles
        style = ttk.Style()
        style.configure("Accent.TButton", font=("Arial", 10, "bold"))

    def load_dl_models(self):
        """Load DL models only"""
        self.status_bar.config(text="Loading DL models...")
        
        # Reset DL status labels
        for model_name in DL_MODEL_PATHS.keys():
            self.model_status_labels[model_name].config(
                text="⏳ Loading...", foreground="orange")
        
        def update_status(model_name, success, error):
            if model_name in self.model_status_labels:
                if success:
                    self.model_status_labels[model_name].config(
                        text="✅ Loaded", foreground="green")
                else:
                    self.model_status_labels[model_name].config(
                        text="❌ Failed", foreground="red")
                    if error:
                        print(f"Failed to load {model_name}: {error}")
        
        threads = self.dl_loader.load_all_models(callback=update_status)
        
        def check_completion():
            loaded_count = len(self.dl_loader.loaded_models)
            total_count = len(DL_MODEL_PATHS)
            
            if loaded_count > 0:
                self.status_bar.config(text=f"{loaded_count}/{total_count} DL models loaded")
            else:
                self.status_bar.config(text="DL models loading failed")
        
        self.root.after(1500, check_completion)

    def load_ml_models(self):
        """Load ML models only"""
        self.status_bar.config(text="Loading ML models...")
        
        # Reset ML status labels
        ml_models = ["SVM", "Random Forest", "Naive Bayes", "Logistic Regression"]
        for model_name in ml_models:
            self.model_status_labels[model_name].config(
                text="⏳ Loading...", foreground="orange")
        
        # Load ML models in a thread
        def load_ml_thread():
            success = self.ml_loader.load_all()
            self.root.after(0, self._update_ml_status, success)
        
        thread = threading.Thread(target=load_ml_thread)
        thread.start()

    def _update_ml_status(self, success):
        """Update ML model status after loading"""
        ml_models = ["SVM", "Random Forest", "Naive Bayes", "Logistic Regression"]
        
        if success:
            loaded_models = self.ml_loader.get_loaded_models()
            for model_name in ml_models:
                if model_name in loaded_models:
                    self.model_status_labels[model_name].config(
                        text="✅ Loaded", foreground="green")
                else:
                    self.model_status_labels[model_name].config(
                        text="❌ Not found", foreground="red")
            
            self.status_bar.config(text=f"{len(loaded_models)} ML models loaded")
        else:
            for model_name in ml_models:
                self.model_status_labels[model_name].config(
                    text="❌ Failed", foreground="red")
            self.status_bar.config(text="ML models loading failed")

    def load_all_models(self):
        """Load both DL and ML models"""
        self.load_dl_models()
        self.load_ml_models()

    def unload_all_models(self):
        """Unload all models"""
        self.dl_loader.loaded_models.clear()
        self.ml_loader.unload_all()
        
        # Reset all status labels
        all_models = list(DL_MODEL_PATHS.keys()) + ["SVM", "Random Forest", "Naive Bayes", "Logistic Regression"]
        for model_name in all_models:
            if model_name in self.model_status_labels:
                self.model_status_labels[model_name].config(
                    text="❌ Not loaded", foreground="red")
            
            # Clear results
            if model_name in self.result_rows:
                self._clear_model_results(model_name)
        
        self._clear_consensus()
        self.status_bar.config(text="All models unloaded")
        messagebox.showinfo("Success", "All models unloaded successfully")

    def _clear_model_results(self, model_name):
        """Clear results for a specific model"""
        if model_name in self.result_rows:
            widgets = self.result_rows[model_name]
            widgets['prediction'].config(text="Not run", foreground="gray")
            widgets['confidence'].config(text="0%")
            for category in EMAIL_CATEGORIES:
                widgets[category].config(text="0%")

    def _clear_consensus(self):
        """Clear consensus results"""
        self.consensus_pred.config(text="N/A")
        self.consensus_conf.config(text="N/A")
        for category in EMAIL_CATEGORIES:
            getattr(self, f"consensus_{category.lower()}").config(text="N/A")

    def browse_file(self):
        """Browse for email files"""
        filetypes = [
            ("Email files", "*.eml"),
            ("Text files", "*.txt"),
            ("All files", "*.*")
        ]
        
        file_path = filedialog.askopenfilename(
            title="Select Email File",
            filetypes=filetypes
        )
        
        if file_path:
            self.current_file_path = file_path
            filename = os.path.basename(file_path)
            file_size = os.path.getsize(file_path) / 1024
            self.file_var.set(f"{filename} ({file_size:.1f} KB)")
            self.status_bar.config(text=f"Selected: {filename}")

    def parse_email(self):
        """Parse the selected email file"""
        if not self.current_file_path:
            messagebox.showerror("Error", "Please select a file first")
            return
        
        try:
            self.text_area.delete(1.0, tk.END)
            self.text_area.insert(1.0, "Parsing...")
            self.status_bar.config(text="Parsing email...")
            
            # Parse based on file type
            if self.current_file_path.lower().endswith('.eml'):
                self.email_text = self.parser.extract_body(self.current_file_path)
            else:
                with open(self.current_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    raw_text = f.read()
                self.email_text = self.parser._clean_text(raw_text)
            
            # Display in text area
            self.text_area.delete(1.0, tk.END)
            self.text_area.insert(1.0, self.email_text)
            
            # Update info
            word_count = len(self.email_text.split())
            char_count = len(self.email_text)
            self.text_info.config(
                text=f"✅ Parsed: {word_count} words, {char_count} chars",
                foreground="green"
            )
            self.status_bar.config(text="Email parsed successfully")
            
        except Exception as e:
            self.text_area.delete(1.0, tk.END)
            self.text_area.insert(1.0, f"Error: {str(e)}")
            self.text_info.config(text="❌ Parse failed", foreground="red")
            self.status_bar.config(text="Parse failed")

    def _classify_all_thread(self):
        """Classify with all loaded models"""
        # Check if any models are loaded
        dl_loaded = len(self.dl_loader.loaded_models) > 0
        ml_loaded = self.ml_loader.loaded
        
        if not dl_loaded and not ml_loaded:
            messagebox.showerror("Error", "Please load models first!")
            return
        
        text = self.text_area.get(1.0, tk.END).strip()
        if not text:
            messagebox.showerror("Error", "Please parse an email first!")
            return
        
        self.status_bar.config(text="Classifying with all models...")
        self.all_results.clear()
        
        # Clear previous results
        all_models = list(DL_MODEL_PATHS.keys()) + ["SVM", "Random Forest", "Naive Bayes", "Logistic Regression"]
        for model_name in all_models:
            self._clear_model_results(model_name)
        self._clear_consensus()
        
        # Create threads for DL models
        threads = []
        for model_name, model_data in self.dl_loader.loaded_models.items():
            thread = threading.Thread(
                target=self._classify_dl_model,
                args=(model_name, model_data, text)
            )
            threads.append(thread)
            thread.start()
        
        # Classify with ML models in current thread (they're fast)
        if ml_loaded:
            for model_name in self.ml_loader.get_loaded_models():
                try:
                    result = self.ml_loader.predict(model_name, text)
                    self.all_results[model_name] = result
                    self.root.after(0, self._update_model_results, model_name, result)
                except Exception as e:
                    result = {
                        'success': False,
                        'error': str(e),
                        'type': 'ml'
                    }
                    self.all_results[model_name] = result
                    self.root.after(0, self._update_model_error, model_name, str(e))
        
        # Start completion checker
        self.root.after(500, self._check_classification_completion)

    def _classify_ml_only(self):
        """Classify with ML models only"""
        if not self.ml_loader.loaded:
            messagebox.showerror("Error", "Please load ML models first!")
            return
        
        text = self.text_area.get(1.0, tk.END).strip()
        if not text:
            messagebox.showerror("Error", "Please parse an email first!")
            return
        
        self.status_bar.config(text="Classifying with ML models...")
        
        # Clear ML model results only
        ml_models = ["SVM", "Random Forest", "Naive Bayes", "Logistic Regression"]
        for model_name in ml_models:
            self._clear_model_results(model_name)
        
        # Classify with each ML model
        self.all_results.clear()
        for model_name in self.ml_loader.get_loaded_models():
            try:
                result = self.ml_loader.predict(model_name, text)
                self.all_results[model_name] = result
                self._update_model_results(model_name, result)
            except Exception as e:
                result = {
                    'success': False,
                    'error': str(e),
                    'type': 'ml'
                }
                self.all_results[model_name] = result
                self._update_model_error(model_name, str(e))
        
        # Calculate ML consensus
        self._calculate_consensus()
        self.status_bar.config(text=f"ML classification complete")

    def _classify_dl_only(self):
        """Classify with DL models only"""
        if not self.dl_loader.loaded_models:
            messagebox.showerror("Error", "Please load DL models first!")
            return
        
        text = self.text_area.get(1.0, tk.END).strip()
        if not text:
            messagebox.showerror("Error", "Please parse an email first!")
            return
        
        self.status_bar.config(text="Classifying with DL models...")
        self.all_results.clear()
        
        # Clear DL model results only
        for model_name in DL_MODEL_PATHS.keys():
            self._clear_model_results(model_name)
        
        # Create threads for DL models
        threads = []
        for model_name, model_data in self.dl_loader.loaded_models.items():
            thread = threading.Thread(
                target=self._classify_dl_model,
                args=(model_name, model_data, text)
            )
            threads.append(thread)
            thread.start()
        
        # Start completion checker
        self.root.after(500, self._check_dl_classification_completion)

    def _classify_dl_model(self, model_name, model_data, text):
        """Classify with a single DL model"""
        try:
            print(f"🔍 {model_name} classifying...")
            
            if model_data['type'] == 'lstm_gru':
                # Use RNN loader's predict method
                label, confidence, probabilities = model_data['loader'].predict(
                    model_data['model'],
                    model_data['tokenizer'],
                    model_data['label_encoder'],
                    text
                )
                
                # Convert probabilities to numpy array
                probabilities = np.array(probabilities, dtype=float)
                
                # Map label if needed
                label = self._get_label_from_string(label, model_data.get('label_encoder'))
                
                result = {
                    'label': label,
                    'confidence': confidence,
                    'probabilities': probabilities,
                    'success': True,
                    'type': 'dl'
                }
                
            else:  # transformer
                # Use transformer loader's predict method
                label, confidence, probabilities = model_data['loader'].predict(
                    model_name,
                    model_data['model'],
                    model_data['tokenizer'],
                    text,
                    model_data.get('label_encoder')
                )
                
                # Ensure probabilities is numpy array
                if not isinstance(probabilities, np.ndarray):
                    probabilities = np.array(probabilities, dtype=float)
                
                # Map label if needed
                label = self._get_label_from_string(label, model_data.get('label_encoder'))
                
                result = {
                    'label': label,
                    'confidence': confidence,
                    'probabilities': probabilities,
                    'success': True,
                    'type': 'dl'
                }
            
            # Store results
            self.all_results[model_name] = result
            
            # Update UI
            self.root.after(0, self._update_model_results, model_name, result)
            
        except Exception as e:
            print(f"Error in {model_name}: {str(e)}\n{traceback.format_exc()}")
            result = {
                'success': False,
                'error': str(e),
                'type': 'dl'
            }
            self.all_results[model_name] = result
            
            self.root.after(0, self._update_model_error, model_name, str(e))

    def _get_label_from_string(self, label, label_encoder):
        """Convert label string to proper category"""
        if label in EMAIL_CATEGORIES:
            return label
        
        # Try to extract from label string
        if isinstance(label, str):
            # Remove Class prefix
            if label.startswith('Class '):
                try:
                    idx = int(label.replace('Class ', ''))
                    if 0 <= idx < len(EMAIL_CATEGORIES):
                        return EMAIL_CATEGORIES[idx]
                except:
                    pass
            
            # Try label encoder inverse mapping
            if label_encoder is not None:
                try:
                    # If label is numeric string
                    if label.isdigit():
                        idx = int(label)
                        if 0 <= idx < len(EMAIL_CATEGORIES):
                            return EMAIL_CATEGORIES[idx]
                    
                    # Check if label exists in label encoder classes
                    if hasattr(label_encoder, 'classes_'):
                        for i, cls in enumerate(label_encoder.classes_):
                            if str(cls) == str(label):
                                if i < len(EMAIL_CATEGORIES):
                                    return EMAIL_CATEGORIES[i]
                except:
                    pass
        
        # Default fallback
        return "Normal"

    def _update_model_results(self, model_name, result):
        """Update UI with model results"""
        if model_name not in self.result_rows:
            return
        
        widgets = self.result_rows[model_name]
        label = result['label']
        confidence = result['confidence'] * 100
        probabilities = result['probabilities']
        
        # Ensure probabilities length matches categories
        if len(probabilities) < len(EMAIL_CATEGORIES):
            # Pad with zeros
            probabilities = np.pad(probabilities, (0, len(EMAIL_CATEGORIES) - len(probabilities)), 'constant')
        elif len(probabilities) > len(EMAIL_CATEGORIES):
            # Truncate
            probabilities = probabilities[:len(EMAIL_CATEGORIES)]
        
        # Normalize probabilities
        probabilities_sum = np.sum(probabilities)
        if probabilities_sum > 0:
            probabilities = probabilities / probabilities_sum
        
        # Set colors based on category
        colors = {
            "Normal": "green",
            "Harassment": "red",
            "Fraudulent": "orange",
            "Suspicious": "blue"
        }
        color = colors.get(label, "black")
        
        # Update prediction
        widgets['prediction'].config(text=label, foreground=color)
        
        # Update confidence
        widgets['confidence'].config(text=f"{confidence:.1f}%")
        
        # Update probabilities
        for i, category in enumerate(EMAIL_CATEGORIES):
            if i < len(probabilities):
                prob_pct = probabilities[i] * 100
                widgets[category].config(text=f"{prob_pct:.1f}%")
        
        self.status_bar.config(text=f"{model_name}: {label} ({confidence:.1f}%)")

    def _update_model_error(self, model_name, error):
        """Update UI with model error"""
        if model_name in self.result_rows:
            widgets = self.result_rows[model_name]
            widgets['prediction'].config(text="Error", foreground="red")
            widgets['confidence'].config(text="N/A")
            for category in EMAIL_CATEGORIES:
                widgets[category].config(text="N/A")

    def _check_classification_completion(self):
        """Check if all DL models have finished classification"""
        dl_loaded = len(self.dl_loader.loaded_models)
        dl_completed = sum(1 for name in self.dl_loader.loaded_models.keys() 
                          if name in self.all_results)
        
        # ML models are already done (they run synchronously)
        ml_loaded = len(self.ml_loader.get_loaded_models())
        ml_completed = sum(1 for name in self.ml_loader.get_loaded_models() 
                          if name in self.all_results)
        
        total_loaded = dl_loaded + ml_loaded
        total_completed = dl_completed + ml_completed
        
        if total_completed >= total_loaded or dl_completed >= dl_loaded:
            # Calculate consensus
            self._calculate_consensus()
            
            success_count = sum(1 for r in self.all_results.values() if r.get('success', False))
            self.status_bar.config(
                text=f"Classification complete: {success_count}/{total_loaded} models succeeded"
            )
        else:
            # Check again after 500ms
            self.root.after(500, self._check_classification_completion)

    def _check_dl_classification_completion(self):
        """Check if DL models have finished classification"""
        dl_loaded = len(self.dl_loader.loaded_models)
        dl_completed = sum(1 for name in self.dl_loader.loaded_models.keys() 
                          if name in self.all_results)
        
        if dl_completed >= dl_loaded:
            # Calculate DL consensus
            self._calculate_consensus()
            
            success_count = sum(1 for r in self.all_results.values() if r.get('success', False))
            self.status_bar.config(
                text=f"DL classification complete: {success_count}/{dl_loaded} models succeeded"
            )
        else:
            self.root.after(500, self._check_dl_classification_completion)

    def _calculate_consensus(self):
        """Calculate consensus across all models"""
        if not self.all_results:
            return
        
        # Collect successful results
        successful_results = []
        for model_name, result in self.all_results.items():
            if result.get('success', False):
                successful_results.append(result)
        
        if not successful_results:
            return
        
        # Average probabilities
        avg_probs = np.zeros(len(EMAIL_CATEGORIES))
        count = 0
        
        for result in successful_results:
            probs = result['probabilities']
            if len(probs) == len(avg_probs):
                avg_probs += probs
                count += 1
        
        if count > 0:
            avg_probs /= count
        
        # Find consensus prediction
        consensus_idx = int(np.argmax(avg_probs))
        consensus_label = EMAIL_CATEGORIES[consensus_idx]
        consensus_conf = avg_probs[consensus_idx] * 100
        
        # Update consensus UI
        colors = {
            "Normal": "green",
            "Harassment": "red",
            "Fraudulent": "orange",
            "Suspicious": "blue"
        }
        color = colors.get(consensus_label, "black")
        
        self.consensus_pred.config(text=consensus_label, foreground=color)
        self.consensus_conf.config(text=f"{consensus_conf:.1f}%")
        
        for i, category in enumerate(EMAIL_CATEGORIES):
            prob_pct = avg_probs[i] * 100 if i < len(avg_probs) else 0
            getattr(self, f"consensus_{category.lower()}").config(text=f"{prob_pct:.1f}%")

    def clear_text(self):
        """Clear the text area"""
        self.text_area.delete(1.0, tk.END)
        self.email_text = ""
        self.text_info.config(text="Text cleared", foreground="gray")
        self.status_bar.config(text="Text cleared")

    def save_results(self):
        """Save classification results"""
        if not self.all_results:
            messagebox.showerror("Error", "No classification results to save")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            initialfile="classification_results.txt"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write("="*70 + "\n")
                    f.write("EMAIL CLASSIFICATION REPORT - ALL MODELS (DL + ML)\n")
                    f.write("="*70 + "\n\n")
                    
                    f.write(f"Date: {datetime.datetime.now()}\n")
                    f.write(f"File: {os.path.basename(self.current_file_path) if self.current_file_path else 'Unknown'}\n")
                    
                    # Count models by type
                    ml_models = [m for m in self.all_results.keys() if m in ["SVM", "Random Forest", "Naive Bayes", "Logistic Regression"]]
                    dl_models = [m for m in self.all_results.keys() if m not in ml_models]
                    
                    f.write(f"Models loaded: {len(self.all_results)} ({len(dl_models)} DL, {len(ml_models)} ML)\n")
                    f.write("\n" + "="*70 + "\n\n")
                    
                    # DL Models results
                    if dl_models:
                        f.write("DEEP LEARNING MODELS:\n")
                        f.write("-"*40 + "\n")
                        for model_name in dl_models:
                            result = self.all_results.get(model_name, {})
                            self._write_model_result(f, model_name, result)
                    
                    # ML Models results
                    if ml_models:
                        f.write("\nMACHINE LEARNING MODELS:\n")
                        f.write("-"*40 + "\n")
                        for model_name in ml_models:
                            result = self.all_results.get(model_name, {})
                            self._write_model_result(f, model_name, result)
                    
                    # Consensus results
                    f.write("\n" + "="*70 + "\n")
                    f.write("CONSENSUS RESULTS\n")
                    f.write("="*70 + "\n\n")
                    
                    consensus_text = self.consensus_pred.cget("text")
                    confidence_text = self.consensus_conf.cget("text")
                    
                    if consensus_text != "N/A":
                        f.write(f"Consensus Prediction: {consensus_text}\n")
                        f.write(f"Average Confidence: {confidence_text}\n\n")
                        
                        f.write("Average Probabilities:\n")
                        for category in EMAIL_CATEGORIES:
                            prob_text = getattr(self, f"consensus_{category.lower()}").cget("text")
                            f.write(f"  {category}: {prob_text}\n")
                
                self.status_bar.config(text=f"Results saved to {os.path.basename(file_path)}")
                messagebox.showinfo("Success", f"Results saved to:\n{file_path}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save: {str(e)}")

    def _write_model_result(self, f, model_name, result):
        """Write individual model result to file"""
        f.write(f"Model: {model_name}\n")
        f.write("-"*30 + "\n")
        
        if result.get('success', False):
            f.write(f"Prediction: {result['label']}\n")
            f.write(f"Confidence: {result['confidence']:.1%}\n")
            f.write("Probabilities:\n")
            
            for i, category in enumerate(EMAIL_CATEGORIES):
                if i < len(result['probabilities']):
                    prob_pct = result['probabilities'][i] * 100
                    f.write(f"  {category}: {prob_pct:.1f}%\n")
        else:
            f.write(f"Status: Failed\n")
            f.write(f"Error: {result.get('error', 'Unknown error')}\n")
        
        f.write("\n")

    def reset(self):
        """Reset everything"""
        self.dl_loader.loaded_models.clear()
        self.ml_loader.unload_all()
        self.current_file_path = None
        self.email_text = ""
        self.all_results.clear()
        
        # Reset UI
        self.file_var.set("No file selected")
        self.text_area.delete(1.0, tk.END)
        self.text_info.config(text="No email loaded", foreground="gray")
        
        all_models = list(DL_MODEL_PATHS.keys()) + ["SVM", "Random Forest", "Naive Bayes", "Logistic Regression"]
        for model_name in all_models:
            if model_name in self.model_status_labels:
                self.model_status_labels[model_name].config(
                    text="❌ Not loaded", foreground="red")
            
            # Clear results
            if model_name in self.result_rows:
                self._clear_model_results(model_name)
        
        self._clear_consensus()
        self.status_bar.config(text="Reset complete")
        messagebox.showinfo("Reset", "All settings have been reset")

if __name__ == "__main__":
    # Suppress TensorFlow warnings
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # Suppress transformers warnings
    os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
    
    root = tk.Tk()
    app = EmailClassifierGUI(root)
    
    # Center window
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'1200x800+{x}+{y}')
    
    root.mainloop()