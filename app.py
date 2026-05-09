from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os
import json
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)  # Allow requests from your GitHub Pages site

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
        radius = data.get('radius', 5)
        lpk_profile = data.get('lpk_profile', {})
        
        # Get elevation data for the area
        elevation_data = get_elevation_grid(lat, lng, radius)
        
        # Analyze with Gemini
        ai_response = analyze_with_gemini(lat, lng, radius, elevation_data, lpk_profile)
        
        return jsonify({'success': True, 'data': ai_response})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

def get_elevation_grid(lat, lng, radius_km):
    """Fetch a grid of elevation points around the LKP"""
    try:
        # Sample 9 points: center + 8 surrounding
        offset_deg = radius_km / 111.0  # rough km to degrees
        points = [
            f"{lat},{lng}",
            f"{lat + offset_deg},{lng}",
            f"{lat - offset_deg},{lng}",
            f"{lat},{lng + offset_deg}",
            f"{lat},{lng - offset_deg}",
            f"{lat + offset_deg},{lng + offset_deg}",
            f"{lat + offset_deg},{lng - offset_deg}",
            f"{lat - offset_deg},{lng + offset_deg}",
            f"{lat - offset_deg},{lng - offset_deg}",
        ]
        locations = "|".join(points)
        url = f"https://api.opentopodata.org/v1/srtm30m?locations={locations}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.json()
        return {"results": [], "note": "Elevation API unavailable"}
    except Exception as e:
        return {"results": [], "error": str(e)}

def analyze_with_gemini(lat, lng, radius, elevation_data, lpk_profile):
    """Send terrain + LPK to Gemini API and request structured polygon zones"""
    
    # Build a clear prompt asking for JSON output
    prompt = f"""You are a Search and Rescue (SAR) terrain analysis expert. Analyze the following situation and generate priority search polygons.

LOCATION: {lat}, {lng} (Last Known Point)
SEARCH RADIUS: {radius} km
LOCATION NAME: {lpk_profile.get('location_name', 'Unknown')}

SUBJECT PROFILE:
- Type: {lpk_profile.get('subject_type', 'Unknown')}
- Age: {lpk_profile.get('age', 'Unknown')}
- Hours missing: {lpk_profile.get('hours_missing', 'Unknown')}
- Terrain type: {lpk_profile.get('terrain', 'Unknown')}

EXPECTED BEHAVIORS:
{chr(10).join('- ' + b for b in lpk_profile.get('behaviors', []))}

ISRID NOTE: {lpk_profile.get('isrid_note', '')}

ELEVATION DATA (sample points around LKP):
{json.dumps(elevation_data, indent=2)[:1500]}

TASK:
Generate 4-6 priority search polygons (zones) based on:
1. Terrain elevation patterns (high ground, drainages, ridgelines)
2. Subject's expected behavior given their profile
3. Likely natural and road boundaries
4. Time elapsed since LKP

Return ONLY a valid JSON object with this exact structure (no markdown, no explanation):
{{
  "zones": [
    {{
      "name": "Zone name (e.g. 'North Ridge - High Ground')",
      "priority": "high" or "medium" or "low",
      "description": "2-3 sentence reasoning based on terrain and subject behavior",
      "bearing_start": 0-360 (compass bearing where zone starts),
      "bearing_end": 0-360 (compass bearing where zone ends),
      "inner_radius_km": number (inner edge from LKP),
      "outer_radius_km": number (outer edge from LKP)
    }}
  ]
}}

Make sure zones cover different sectors and reflect the subject's likely travel patterns."""
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.4,
            "maxOutputTokens": 2048
        }
    }
    
    response = requests.post(url, json=body, timeout=30)
    raw_response = response.json()
    
    # Try to parse the structured zones from the response
    try:
        text = raw_response['candidates'][0]['content']['parts'][0]['text']
        # Strip markdown code fences if present
        text = text.replace('```json', '').replace('```', '').strip()
        parsed = json.loads(text)
        
        # Convert bearing-based zones into actual polygon coordinates
        if 'zones' in parsed:
            for zone in parsed['zones']:
                if 'bearing_start' in zone and 'bearing_end' in zone:
                    zone['polygon'] = generate_polygon_coords(
                        lat, lng,
                        zone.get('inner_radius_km', radius * 0.2),
                        zone.get('outer_radius_km', radius * 0.85),
                        zone['bearing_start'],
                        zone['bearing_end']
                    )
        
        return parsed
    except Exception as e:
        # Return raw response if parsing fails - frontend will fallback
        return raw_response

def generate_polygon_coords(lat, lng, inner_km, outer_km, bearing_start, bearing_end):
    """Generate polygon coordinates as [[lat, lng], ...] for a wedge shape"""
    import math
    R = 6371  # Earth radius in km
    points = []
    
    # Normalize bearings
    if bearing_end < bearing_start:
        bearing_end += 360
    
    # Inner arc
    bearing = bearing_start
    while bearing <= bearing_end:
        p = point_from_bearing(lat, lng, inner_km, bearing % 360)
        points.append(p)
        bearing += 5
    
    # Outer arc (reverse)
    bearing = bearing_end
    while bearing >= bearing_start:
        p = point_from_bearing(lat, lng, outer_km, bearing % 360)
        points.append(p)
        bearing -= 5
    
    # Close polygon
    if points:
        points.append(points[0])
    
    return points

def point_from_bearing(lat, lng, dist_km, bearing_deg):
    """Calculate a point given start, distance, and bearing"""
    import math
    R = 6371
    d = dist_km / R
    brng = math.radians(bearing_deg)
    lat_r = math.radians(lat)
    lng_r = math.radians(lng)
    
    p_lat = math.asin(math.sin(lat_r) * math.cos(d) + math.cos(lat_r) * math.sin(d) * math.cos(brng))
    p_lng = lng_r + math.atan2(
        math.sin(brng) * math.sin(d) * math.cos(lat_r),
        math.cos(d) - math.sin(lat_r) * math.sin(p_lat)
    )
    
    return [math.degrees(p_lat), math.degrees(p_lng)]

if __name__ == '__main__':
    app.run()