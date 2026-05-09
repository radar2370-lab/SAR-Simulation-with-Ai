from flask import Flask, request, jsonify
import requests
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

@app.route('/analyze-sar', methods=['POST'])
def analyze_sar():
    try:
        data = request.json
        lat = data.get('lat')
        lng = data.get('lng')
        radius = data.get('radius')
        lpk_profile = data.get('lpk_profile')
        
        elevation_data = get_elevation_grid(lat, lng, radius)
        search_grids = analyze_with_gemini(elevation_data, lpk_profile)
        
        return jsonify({'success': True, 'data': search_grids})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

def get_elevation_grid(lat, lng, radius):
    url = f"https://api.opentopodata.org/v1/test-dataset?locations={lat},{lng}"
    response = requests.get(url)
    return response.json()

def analyze_with_gemini(elevation_data, lpk_profile):
    prompt = f"Analyze this terrain for SAR: {elevation_data} with LPK: {lpk_profile}"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    body = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    response = requests.post(url, json=body)
    return response.json()

if __name__ == '__main__':
    app.run()