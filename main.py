from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import numpy as np
from PIL import Image
from io import BytesIO
import requests

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
    data = request.json
    cleaned_text = stemming(data.get("title", ""))

    if not cleaned_text.strip():
        return {"error": "Input is empty after preprocessing."}

    try:
        vector_input = vectorizer.transform([cleaned_text])
        prediction = lr.predict(vector_input)

        result = "Fake News" if prediction[0] == 1 else "Real News"
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
    data = request.json
    input_mail = data.get("message", "")

    if not input_mail.strip():
        return {"error": "Input is empty after preprocessing."}

    try:
        feature_extraction = spamvectorizer.transform([input_mail])
        prediction = spamModel.predict(feature_extraction)

        result = "Spam Mail" if prediction[0] == 0 else "Not Spam"
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
    data = request.json
    input_url = data.get("link", "")

    if not input_url.strip():
        return {"error": "Input is empty after preprocessing."}

    try:
        feature_extraction = url_vectorizer.transform([input_url])
        prediction = phishing_model.predict(feature_extraction)

        result = "Safe Url" if prediction[0] == 0 else "Phishing Url"
        return {"prediction": result}
    except Exception as e:
        return {"error": str(e)}
    
# models for breast cancer prediction
breast_cancer_model_file_id = '1Mp9GDNnVx0wbJ5fM32tmYUryKQSmZbMq'
breast_cancer_model = load_model_from_drive(breast_cancer_model_file_id)

standard_scaler_file_id = '1p5BdQSy7WbMip3AhT44acbOQTOnj0vzV'
standard_scaler = load_model_from_drive(standard_scaler_file_id)

@app.route("/breastcancerpredict", methods=["POST"])
def predict_cancer():
    data = request.json
    try:
        input_features = [
            data.get("radius_mean"), data.get("texture_mean"), data.get("perimeter_mean"), data.get("area_mean"),
            data.get("smoothness_mean"), data.get("compactness_mean"), data.get("concavity_mean"),
            data.get("concave_points_mean"), data.get("symmetry_mean"), data.get("fractal_dimension_mean"),
            data.get("radius_se"), data.get("texture_se"), data.get("perimeter_se"), data.get("area_se"),
            data.get("smoothness_se"), data.get("compactness_se"), data.get("concavity_se"),
            data.get("concave_points_se"), data.get("symmetry_se"), data.get("fractal_dimension_se"),
            data.get("radius_worst"), data.get("texture_worst"), data.get("perimeter_worst"), data.get("area_worst"),
            data.get("smoothness_worst"), data.get("compactness_worst"), data.get("concavity_worst"),
            data.get("concave_points_worst"), data.get("symmetry_worst"), data.get("fractal_dimension_worst")
        ]
        starndarized = standard_scaler.transform([input_features])
        prediction = breast_cancer_model.predict(starndarized)
        prediction_labels = [np.argmax(prediction)]

        result = "Malignant (Cancerous)" if prediction_labels[0] == 0 else "Benign (Non-cancerous)"
        return {"prediction": result}
    except Exception as e:
        return {"error": str(e)}
    

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
