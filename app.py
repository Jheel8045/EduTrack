from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ✅ Load model and scaler from backend folder
model = joblib.load(os.path.join(BASE_DIR, "student_performance_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

@app.route('/')
def home():
    return "🎓 EduTrack ML API is running!"

# ✅ Single student prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = np.array([list(data.values())])
        scaled_features = scaler.transform(features)
        prediction = model.predict(scaled_features)[0]
        return jsonify({'predicted_grade_class': int(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)})

# ✅ CSV upload + bulk prediction
@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files['file']
        if not file.filename.endswith('.csv'):
            return jsonify({"error": "Only CSV files are allowed"}), 400

        # Save temporarily
        upload_folder = "uploads"
        os.makedirs(upload_folder, exist_ok=True)
        filepath = os.path.join(upload_folder, "uploaded.csv")
        file.save(filepath)

        # Read file
        df = pd.read_csv(filepath)

        # Check for required columns
        required_columns = [
            'Age', 'Gender', 'Ethnicity', 'ParentalEducation', 'StudyTimeWeekly',
            'Absences', 'Tutoring', 'ParentalSupport', 'Extracurricular',
            'Sports', 'Music', 'Volunteering', 'GPA'
        ]
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            return jsonify({"error": f"Missing columns: {missing}"}), 400

        # Scale + Predict
        features = df[required_columns].values
        scaled_features = scaler.transform(features)
        predictions = model.predict(scaled_features)

        # Add results
        df['Predicted_GradeClass'] = predictions

        # Save results
        output_folder = "outputs"
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, "predicted_output.csv")
        df.to_csv(output_path, index=False)

        return jsonify({
            "message": "✅ Predictions generated successfully!",
            "download_url": f"http://127.0.0.1:5000/download/predicted_output.csv"
        })
    
    except Exception as e:
        return jsonify({"error": str(e)})

# ✅ Download predicted file
@app.route("/download/<path:filename>", methods=["GET"])
def download_file(filename):
    try:
        output_path = os.path.join("outputs", filename)
        if not os.path.exists(output_path):
            return jsonify({"error": "File not found"}), 404
        return send_file(output_path, as_attachment=True)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
