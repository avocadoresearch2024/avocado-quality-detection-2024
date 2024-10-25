from flask import Flask, request, jsonify
from flask_cors import CORS 
from Functions.GrowthDetection import *
from Functions.PestDiseaseIdentification import *
from Functions.PricePrediction import *
from Functions.QualityIdentification import *
import json

app = Flask(__name__)
CORS(app)

@app.route('/growth-detection', methods=['POST'])
def detect_growth():
    try:
        if 'image_path' not in request.files:
            return jsonify({"error": "Image not found"}), 400
        
        image_path = request.files['image_path']
        month = request.form.get('month')

        if not image_path or not month :
            return jsonify({"error": "Invalid request, please provide image and month"}), 400

        data =  growthDetection(image_path, month)
        return jsonify({'status_code':200,'success':True,'data': data}),200
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e),'status_code':500}),500

@app.route('/pest-disease', methods=['POST'])
def classify_disease():
    try:
        if 'image_path' not in request.files:
            return jsonify({"error": "Image not found"}), 400
        
        image_path = request.files['image_path']

        if not image_path :
            return jsonify({"error": "Invalid request, please provide image"}), 400
        
        data =  pestDiseaseIdentification(image_path)
        return jsonify({'status_code':200,'success':True,'data': data}),200
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e),'status_code':500}),500

@app.route('/price-prediction', methods=['POST'])
def predict_price():
    try:
        body = request.get_json() 
        timePeriod = body['time_period']
        data = predictPrice( timePeriod)
        data =  json.dumps(data)
        return jsonify({'status_code':200,'success':True,'data': data}),200

    except Exception as e:
        return jsonify({"success": False, "error": str(e),'status_code':500}),500

@app.route('/quality-identification', methods=['POST'])
def quality():
    try:
        if 'image_path' not in request.files:
            return jsonify({"error": "Image not found"}), 400
        
        image_paths = request.files.getlist("image_path")

        if len(image_paths) < 1 :
            return jsonify({"error": "Invalid request, please provide atleast one image"}), 400
        
        data =  qualityIdentification(image_paths)
        return jsonify({'status_code':200,'success':True,'data': data}),200
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e),'status_code':500}),500


if __name__ == '__main__':
    app.run(debug=True)
