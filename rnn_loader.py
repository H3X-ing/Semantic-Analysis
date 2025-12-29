import tensorflow as tf
import joblib
import numpy as np

class RNNModelLoader:
    def load_model(self, path):
        model = tf.keras.models.load_model(f"{path}/model.h5")
        tokenizer = joblib.load(f"{path}/tokenizer.pkl")
        label_encoder = joblib.load(f"{path}/label_encoder.pkl")
        return model, tokenizer, label_encoder

    def predict(self, model, tokenizer, label_encoder, text, max_len=300):
        seq = tokenizer.texts_to_sequences([text])
        padded = tf.keras.preprocessing.sequence.pad_sequences(
            seq, maxlen=max_len
        )

        probs = model.predict(padded, verbose=0)[0]
        idx = int(np.argmax(probs))
        confidence = float(probs[idx])
        label = label_encoder.inverse_transform([idx])[0]

        return label, confidence, probs.tolist()
