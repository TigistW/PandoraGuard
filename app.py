from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import joblib

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can specify a list of allowed origins or use "*" for all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class LoadModel:
    def load_cnn_model(self, cnn_model_path):
        cnn_model = joblib.load(cnn_model_path)
        return cnn_model
    def load_tfidf_vectorizer(self, tfidf_file_path):
        tfidf_vectorizer = joblib.load(tfidf_file_path)
        return tfidf_vectorizer
    def load_label_encoder(self, label_encoder_path):
        label_encoder = joblib.load(label_encoder_path)
        return label_encoder

# Define input data model
class InputText(BaseModel):
    text: str

@app.post("/predict")
def predict_hate_speech(input_text: InputText):
    # Preprocess the input text
    preprocessed_text = preprocess_input(input_text.text)
    
    model_loader = LoadModel()
    tfidf_vectorizer_path = "tfidf_vectorizer.pkl"
    model_path = "cnn_model.pkl"
    label_encoder_path = "tfidf_vectorizer.pkl"
    
    
    cnn_model = model_loader.load_cnn_model(model_path)
    tfidf_vectorizer = model_loader.load_tfidf_vectorizer(tfidf_vectorizer_path)
    label_encoder = model_loader.load_label_encoder(label_encoder_path)
    
    print("Loaded succusfully")
    

    # Vectorize the text using TF-IDF vectorizer
    vectorized_text = tfidf_vectorizer.transform([preprocessed_text]).toarray()

    custom_predictions = cnn_model.predict(vectorized_text)
    
    custom_predictions = (custom_predictions> 0.5).astype(int).flatten()
    

    return f"text: {input_text.text}  predicted_class: {custom_predictions}"

def preprocess_input(text):
    # Implement your text preprocessing logic here
    # This may include lowercasing, removing stopwords, etc.
    # Should be consistent with the preprocessing used during training
    return text

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
