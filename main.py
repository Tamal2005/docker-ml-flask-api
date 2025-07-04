from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import numpy as np
from PIL import Image
from io import BytesIO
import requests
import sys

sys.stdout.reconfigure(line_buffering=True)

nltk.download('stopwords')

# Initialize Flask app
app = Flask(__name__)
CORS(app)


def load_model_from_drive(file_id):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(url)
    if response.status_code == 200:
        return joblib.load(BytesIO(response.content))
    else:
        raise Exception(f"Failed to download model from Google Drive. Status code: {response.status_code}")


# Preprocessing
port_stem = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = content.lower().split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if word not in stopwords.words('english')]
    return ' '.join(stemmed_content)

# Root route (for testing)
@app.route("/", methods=["GET"])
def home():
    return {"message": " Machine Learning API"}

# models for fake news prediction
lr_file_id = '1J9rL9Y0l2LHlCrODI8whsWCut-9rNTtI'
lr = load_model_from_drive(lr_file_id)

vectorizer_file_id = '1nK5vdINYUVjsfZCu-Yp1j-KGXcHG1ZW5'
vectorizer = load_model_from_drive(vectorizer_file_id)

# Fake News Prediction endpoint
@app.route("/fakenewspredict", methods=["POST"])
def predict_news():

    try:
        data = request.get_json()
        print(data, flush=True)
        cleaned_text = stemming(data.get("title", ""))
        
        vector_input = vectorizer.transform([cleaned_text])
        prediction = lr.predict(vector_input)

        result = "Fake News" if prediction[0] == 1 else "Real News"
        print(result, flush=True)
        return {"prediction": result}
    except Exception as e:
        return {"error": str(e)}

# models for spam mail prediction
spam_Model_file_id = '1Mp9GDNnVx0wbJ5fM32tmYUryKQSmZbMq'
spamModel = load_model_from_drive(spam_Model_file_id)

spam_vector_file_id = '1p5BdQSy7WbMip3AhT44acbOQTOnj0vzV'
spamvectorizer = load_model_from_drive(spam_vector_file_id)

# Spam Mails Prediction endpoint    
@app.route("/spammailpredict", methods=["POST"])
def predict_mails():
    try:
        data = request.get_json()
        input_mail = data.get("message", "").strip()
        print(input_mail, flush=True)

        feature_extraction = spamvectorizer.transform([input_mail])
        
        prediction = spamModel.predict(feature_extraction)

        result = "Spam Mail" if prediction[0] == 0 else "Not Spam"
        print(result, flush=True)
        return {"prediction": result}
    except Exception as e:
        return {"error": str(e)}
    
# Load model only once at startup
masks_model_file_id = "1d6EdeggM7nGPwkEc8Pt9CDRE3J7vXmro"
masksModel = load_model_from_drive(masks_model_file_id)

# Face Mask Prediction endpoint    
@app.route("/facemaskspredict", methods=["POST"])
def predict_facemask():
    try:
        image_file = request.files['image']
        # Read image bytes
        image_bytes = image_file.read()

        # Convert to image array using PIL + OpenCV
        img = Image.open(BytesIO(image_bytes)).convert("")
        img = img.resize((128, 128))
        img_np = np.array(img)
        img_np = img_np / 255.0
        img_np = np.reshape(img_np, (1, 128, 128, 3))

        prediction = masksModel.predict(img_np)

        prediction_label = np.argmax(prediction)

        result = "The person in the image is not wearing a mask" if prediction_label == 0 else "The person in the image is wearing a mask"
        return {"prediction": result}
    except Exception as e:
        return {"error": str(e)}
    

phishing_model_file_id = '1ikdw85kNEmIDW4zaPUcWEXpkSvHNOuZQ'
phishing_model = load_model_from_drive(phishing_model_file_id)
url_vectorizer = joblib.load('./model/phishingUrlModel/url_Vectorizer.pkl')

# Phishing Url Prediction endpoint    
@app.route("/phishingurlpredict", methods=["POST"])
def predict_Urls():
    try:
        data = request.get_json()
        input_url = data.get("link", "").strip()
        print(input_url, flush=True)
        
        feature_extraction = url_vectorizer.transform([input_url])
        
        prediction = phishing_model.predict(feature_extraction)
        

        result = "Safe Url" if prediction[0] == 0 else "Phishing Url"
        print(result)
        return {"prediction": result}
    except Exception as e:
        return {"error": str(e)}

    

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
