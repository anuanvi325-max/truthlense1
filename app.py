from flask import Flask, render_template, request, jsonify
import pickle
import os
import cv2
import pytesseract

app = Flask(__name__)

# LOAD FAKE NEWS MODEL
model = pickle.load(open('models/model.pkl', 'rb'))
vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))

# TESSERACT PATH
if os.name == "nt":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
else:
    pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

def predict_news(text):
    vector = vectorizer.transform([text])
    pred = model.predict(vector)
    return "REAL" if pred[0] == 1 else "FAKE"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_text', methods=['POST'])
def predict_text():
    news = request.form['news']
    result = predict_news(news)
    return jsonify({"prediction": result})

@app.route('/predict_image', methods=['POST'])
def predict_image():

    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"})

    file = request.files['image']

    if file.filename == '':
        return jsonify({"error": "No selected file"})

    if not os.path.exists("uploads"):
        os.makedirs("uploads")

    path = os.path.join("uploads", file.filename)
    file.save(path)

    # OCR
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)

    # Fake News
    news_result = predict_news(text)

    return jsonify({
        "news_result": news_result,
        "extracted_text": text
    })

if __name__ == '__main__':
    app.run(debug=True)