from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

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
        with open(cnn_model_path, "rb") as file:
            cnn_model = pickle.load(file)
        return cnn_model
    def load_tfidf_vectorizer(self, tfidf_file_path):
        with open(tfidf_file_path, "rb") as file:
            tfidf_vectorizer = pickle.load(file)
        return tfidf_vectorizer
    def load_label_encoder(self, label_encoder_path):
        with open(label_encoder_path, "rb") as file:
            label_encoder = pickle.load(file)
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
    

    # Vectorize the text using TF-IDF vectorizer
    vectorized_text = tfidf_vectorizer.transform([preprocessed_text])

    # Make predictions using the CNN model
    cnn_input = np.array(vectorized_text.toarray())  # Convert to NumPy array
    prediction_prob = cnn_model.predict(cnn_input)[0][0]

    # Map prediction probability to class label using label encoder
    predicted_class = label_encoder.inverse_transform([int(round(prediction_prob))])[0]

    return {"text": input_text.text, "predicted_class": predicted_class, "prediction_prob": prediction_prob}

def preprocess_input(text):
    # Implement your text preprocessing logic here
    # This may include lowercasing, removing stopwords, etc.
    # Should be consistent with the preprocessing used during training
    return text

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
