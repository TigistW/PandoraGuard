# Hate Speech Detection API

This is a simple FastAPI-based RESTful API for hate speech detection. It utilizes a Convolutional Neural Network (CNN) model trained for this purpose. The API takes text input, preprocesses it, and then predicts whether it contains hate speech or not.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/TigistW/PandoraGuard.git
   ```

2. Navigate to the project directory:

   ```bash
   cd PandoraGuard
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Start the API server:

   ```bash
   uvicorn main:app --reload
   ```

2. Once the server is up and running, you can send POST requests to `http://127.0.0.1:8000/predict` with the text you want to analyze. The text should be sent in the request body in JSON format with the key `text`.

   Example using cURL:

   ```bash
   curl -X 'POST' \
     'http://127.0.0.1:8000/predict' \
     -H 'accept: application/json' \
     -H 'Content-Type: application/json' \
     -d '{
     "text": "Put your text here in amharic"
   }'
   ```

3. The API will respond with JSON containing the predicted class (`0` for non-hate speech, `1` for hate speech) and the confidence level of the prediction.

## Preprocessing

Text preprocessing is an essential step in natural language processing tasks. The API expects input text to be preprocessed before making predictions. If you're training your own model or using this API with data not preprocessed similarly to the training data, you should preprocess your text accordingly. You can customize the `preprocess_input()` function in the `main.py` file to fit your preprocessing requirements.

## Model and Vectorizer

The API uses a pre-trained Convolutional Neural Network (CNN) model for hate speech detection. The model file (`cnn_model.pkl`) and TF-IDF vectorizer file (`tfidf_vectorizer.pkl`) should be placed in the project directory. Ensure these files are present and accessible to the API.

## CORS Configuration

Cross-Origin Resource Sharing (CORS) is enabled by default to allow requests from any origin. You can modify the CORS configuration in the `main.py` file as needed.

## Feedback and Contributions

Feedback, bug reports, and contributions are welcome. If you encounter any issues or have suggestions for improvements, please create an issue on GitHub or submit a pull request.

## Disclaimer

This API is provided for educational and demonstration purposes only. It may not be suitable for production use without further testing and customization. The developers are not responsible for any misuse or consequences resulting from the use of this software.# Hate Speech Detection API