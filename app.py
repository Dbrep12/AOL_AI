from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import pickle
import os
import requests
from datetime import datetime

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Global variables
scaler = None
model = None
feature_names = None
model_metrics = {}

# City to ADM4 code mapping
CITY_CODES = {
    'jakarta': '31.71.03.1001',
    'bekasi': '32.16.09.2007',
    'cikarang': '32.16.09.2007',
    'bandung': '33.74.05.1001',
    'surabaya': '35.78.03.1001',
    'semarang': '33.74.04.1001',
    'yogyakarta': '34.71.02.1001',
    'malang': '35.73.05.1001',
    'bogor': '32.01.05.2001',
    'tangerang': '36.71.06.1001',
    'depok': '32.76.04.1001',
    'medan': '12.71.01.1001',
    'palembang': '16.71.04.1001',
    'makassar': '73.71.04.1001',
    'denpasar': '51.71.02.1001',
}

BMKG_API_URL = "https://api.bmkg.go.id/publik/prakiraan-cuaca"

def get_city_code(city_name):
    """Convert city name to ADM4 code"""
    city_lower = city_name.lower().strip()
    return CITY_CODES.get(city_lower, '32.16.09.2007')  # Default to Cikarang

def fetch_bmkg_data(city_name):
    """Fetch real weather data from BMKG API using city name"""
    try:
        adm4_code = get_city_code(city_name)
        url = f"{BMKG_API_URL}?adm4={adm4_code}"
        
        print(f"\n🌐 Fetching BMKG data for {city_name} (code: {adm4_code})...")
        
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        location = data.get('lokasi', {})
        weather_data = data.get('data', [{}])[0].get('cuaca', [[]])[0]
        
        if not weather_data:
            print("❌ No weather data found")
            return None
        
        latest = weather_data[0] if weather_data else {}
        
        bmkg_params = {
            'location': {
                'provinsi': location.get('provinsi', 'Unknown'),
                'kotkab': location.get('kotkab', 'Unknown'),
                'kecamatan': location.get('kecamatan', 'Unknown'),
                'desa': location.get('desa', 'Unknown'),
                'lat': location.get('lat', 0),
                'lon': location.get('lon', 0)
            },
            'weather': {
                'temperature': latest.get('t', 27),
                'humidity': latest.get('hu', 75),
                'wind_speed': latest.get('ws', 5),
                'wind_direction': latest.get('wd', 'N'),
                'weather_desc': latest.get('weather_desc', 'Cerah'),
                'total_cloud_cover': latest.get('tcc', 50),
                'rainfall_prob': latest.get('tp', 0),
                'visibility': latest.get('vs', 10000),
                'datetime': latest.get('local_datetime', ''),
            }
        }
        
        print(f"✅ Got weather data: {bmkg_params['weather']['weather_desc']}, {bmkg_params['weather']['temperature']}°C")
        return bmkg_params
        
    except Exception as e:
        print(f"❌ Error fetching BMKG data: {e}")
        return None

def estimate_flood_parameters_from_bmkg(bmkg_data, city_name):
    """Convert BMKG weather data to flood prediction parameters"""
    if not bmkg_data:
        return None
    
    weather = bmkg_data['weather']
    location = bmkg_data['location']
    
    cloud_cover = weather['total_cloud_cover']
    rain_prob = weather['rainfall_prob']
    
    # Rainfall intensity estimation
    if rain_prob > 70 and cloud_cover > 80:
        rainfall_intensity = np.random.uniform(25, 50)
    elif rain_prob > 40 and cloud_cover > 60:
        rainfall_intensity = np.random.uniform(12, 28)
    elif rain_prob > 20:
        rainfall_intensity = np.random.uniform(3, 15)
    else:
        rainfall_intensity = np.random.uniform(0, 4)
    
    rainfall_duration = (rain_prob / 100) * 8
    soil_moisture = min(weather['humidity'] * 0.8 + rain_prob * 0.2, 100)
    
    if rain_prob > 60:
        previous_24h_rainfall = np.random.uniform(50, 130)
    elif rain_prob > 30:
        previous_24h_rainfall = np.random.uniform(20, 60)
    else:
        previous_24h_rainfall = np.random.uniform(0, 25)
    
    river_level = min((rain_prob / 100) * 6 + (rainfall_intensity / 100) * 2, 8)
    
    # City-specific parameters with AREA SIZE and POPULATION DENSITY
    city_params = {
        'jakarta': {
            'soil_height': 5, 'drainage': 45, 'urbanization': 90, 'deforestation': 70, 
            'slope': 1, 'distance': 0.8, 'sea_level': 0.5,
            'area_size': 664.0,  # Jakarta area in km²
            'population_density': 15234,  # People per km²
            'area_type': 'Province Capital'
        },
        'bekasi': {
            'soil_height': 15, 'drainage': 50, 'urbanization': 75, 'deforestation': 60, 
            'slope': 2, 'distance': 1.2, 'sea_level': 0.4,
            'area_size': 210.0,
            'population_density': 13500,
            'area_type': 'City'
        },
        'cikarang': {
            'soil_height': 15, 'drainage': 55, 'urbanization': 70, 'deforestation': 55, 
            'slope': 2, 'distance': 1.0, 'sea_level': 0.3,
            'area_size': 85.0,
            'population_density': 8500,
            'area_type': 'District'
        },
        'bandung': {
            'soil_height': 120, 'drainage': 65, 'urbanization': 75, 'deforestation': 45, 
            'slope': 15, 'distance': 2.5, 'sea_level': 0.1,
            'area_size': 167.0,
            'population_density': 14720,
            'area_type': 'City'
        },
        'surabaya': {
            'soil_height': 8, 'drainage': 50, 'urbanization': 85, 'deforestation': 65, 
            'slope': 1, 'distance': 0.6, 'sea_level': 0.6,
            'area_size': 350.0,
            'population_density': 8464,
            'area_type': 'City'
        },
        'semarang': {
            'soil_height': 10, 'drainage': 55, 'urbanization': 70, 'deforestation': 50, 
            'slope': 3, 'distance': 0.9, 'sea_level': 0.5,
            'area_size': 373.0,
            'population_density': 4456,
            'area_type': 'City'
        },
        'yogyakarta': {
            'soil_height': 80, 'drainage': 60, 'urbanization': 65, 'deforestation': 40, 
            'slope': 8, 'distance': 2.0, 'sea_level': 0.1,
            'area_size': 32.5,
            'population_density': 12699,
            'area_type': 'City'
        },
        'bogor': {
            'soil_height': 190, 'drainage': 60, 'urbanization': 60, 'deforestation': 50, 
            'slope': 12, 'distance': 1.8, 'sea_level': 0.1,
            'area_size': 118.0,
            'population_density': 9038,
            'area_type': 'City'
        },
    }
    
    city_lower = city_name.lower().strip()
    city_data = city_params.get(city_lower, city_params['cikarang'])
    
    params = {
        'soil_height': city_data['soil_height'],
        'rainfall_intensity': round(rainfall_intensity, 1),
        'rainfall_duration': round(rainfall_duration, 1),
        'river_water_level': round(river_level, 1),
        'drainage_capability': city_data['drainage'],
        'urbanization_index': city_data['urbanization'],
        'deforestation_index': city_data['deforestation'],
        'sea_level_rise': city_data['sea_level'],
        'soil_moisture': round(soil_moisture, 1),
        'terrain_slope': city_data['slope'],
        'distance_to_river': city_data['distance'],
        'previous_24h_rainfall': round(previous_24h_rainfall, 1),
        'area_size': city_data['area_size'],  # NEW
        'population_density': city_data['population_density'],  # NEW
        'area_type': city_data['area_type'],  # NEW
        'bmkg_temperature': weather['temperature'],
        'bmkg_humidity': weather['humidity'],
        'bmkg_weather': weather['weather_desc'],
        'location_name': f"{location['desa']}, {location['kecamatan']}, {location['kotkab']}"
    }
    
    return params

def generate_training_data(n_samples=8000):
    """Generate realistic training data based on Indonesian climate patterns"""
    np.random.seed(42)
    
    data = {
        'soil_height': np.random.normal(30, 40, n_samples).clip(-20, 300),
        'rainfall_intensity': np.random.gamma(2, 12, n_samples).clip(0, 100),
        'rainfall_duration': np.random.gamma(2, 3, n_samples).clip(0, 24),
        'previous_24h_rainfall': np.random.gamma(2.5, 20, n_samples).clip(0, 300),
        'river_water_level': np.random.gamma(2, 1.5, n_samples).clip(0, 10),
        'distance_to_river': np.random.exponential(2, n_samples).clip(0.1, 10),
        'drainage_capability': np.random.beta(3, 2.5, n_samples) * 100,
        'urbanization_index': np.random.beta(2, 2, n_samples) * 100,
        'deforestation_index': np.random.beta(2, 2.5, n_samples) * 100,
        'sea_level_rise': np.random.gamma(1.5, 0.25, n_samples).clip(0, 3),
        'soil_moisture': np.random.beta(4, 2, n_samples) * 100,
        'terrain_slope': np.random.gamma(2, 5, n_samples).clip(0, 45),
        'area_size': np.random.gamma(3, 15, n_samples).clip(0.1, 500),  # NEW: Area size in km²
        'population_density': np.random.gamma(2, 2000, n_samples).clip(10, 20000),  # NEW: People per km²
    }
    
    df = pd.DataFrame(data)
    
    # Add correlations
    df['soil_moisture'] = df['soil_moisture'] + (df['rainfall_intensity'] * 0.25)
    df['soil_moisture'] = df['soil_moisture'].clip(0, 100)
    
    df['drainage_capability'] = df['drainage_capability'] - (df['urbanization_index'] * 0.18)
    df['drainage_capability'] = df['drainage_capability'].clip(0, 100)
    
    # Population density affects drainage (more people = worse drainage)
    df['drainage_capability'] = df['drainage_capability'] - (df['population_density'] / 20000 * 15)
    df['drainage_capability'] = df['drainage_capability'].clip(0, 100)
    
    coastal_mask = df['soil_height'] < 20
    df.loc[coastal_mask, 'river_water_level'] += df.loc[coastal_mask, 'sea_level_rise'] * 1.2
    df['river_water_level'] = df['river_water_level'].clip(0, 10)
    
    # Calculate flood risk with area size consideration
    flood_risk = (
        (df['rainfall_intensity'] * 0.18) +
        (df['rainfall_duration'] * 0.10) +
        (df['soil_moisture'] * 0.09) +
        (df['river_water_level'] * 0.13) +
        (df['previous_24h_rainfall'] * 0.09) +
        ((100 - df['drainage_capability']) * 0.11) +
        (df['deforestation_index'] * 0.05) +
        (df['urbanization_index'] * 0.07) +
        ((10 - df['distance_to_river']) * 0.04) +
        ((45 - df['terrain_slope']) * 0.02) +
        (df['sea_level_rise'] * 0.01) +
        (np.log1p(df['area_size']) * 0.06) +  # NEW: Larger areas = higher cumulative risk
        (df['population_density'] / 20000 * 0.05)  # NEW: Higher density = more vulnerable
    )
    
    # Interaction effects
    interaction_1 = (df['rainfall_intensity'] / 100) * (df['soil_moisture'] / 100) * 6
    interaction_2 = np.where(df['soil_height'] < 10, df['river_water_level'] * 2.5, 0)
    interaction_3 = ((100 - df['drainage_capability']) / 100) * (df['urbanization_index'] / 100) * 4
    # NEW: Area size amplifies risk when combined with poor drainage
    interaction_4 = (df['area_size'] / 100) * ((100 - df['drainage_capability']) / 100) * 2
    
    flood_risk = flood_risk + interaction_1 + interaction_2 + interaction_3 + interaction_4
    
    threshold = np.percentile(flood_risk, 68)
    df['flash_flood'] = (flood_risk > threshold).astype(int)
    
    return df

def train_model():
    """Train the Random Forest model"""
    global scaler, model, feature_names, model_metrics
    
    print("\n" + "="*70)
    print("TRAINING FLOOD PREDICTION MODEL")
    print("="*70)
    
    df = generate_training_data(n_samples=8000)
    
    flood_count = df['flash_flood'].sum()
    print(f"\n📊 Flood cases: {flood_count} ({flood_count/len(df)*100:.1f}%)")
    
    X = df.drop('flash_flood', axis=1)
    y = df['flash_flood']
    feature_names = X.columns.tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("🌲 Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=18,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    
    train_accuracy = model.score(X_train_scaled, y_train)
    test_accuracy = model.score(X_test_scaled, y_test)
    
    print(f"\n📈 Training Accuracy: {train_accuracy:.2%}")
    print(f"📈 Testing Accuracy:  {test_accuracy:.2%}")
    
    model_metrics = {
        'train_accuracy': float(train_accuracy),
        'test_accuracy': float(test_accuracy),
        'n_samples': len(df),
        'n_features': len(feature_names)
    }
    
    with open('flood_model_bmkg.pkl', 'wb') as f:
        pickle.dump({
            'model': model,
            'scaler': scaler,
            'features': feature_names,
            'metrics': model_metrics
        }, f)
    
    print("\n✅ Model training complete!")
    print("="*70 + "\n")

def load_model():
    """Load existing model or train new one"""
    global scaler, model, feature_names, model_metrics
    
    if os.path.exists('flood_model_bmkg.pkl'):
        print("📂 Loading model...")
        try:
            with open('flood_model_bmkg.pkl', 'rb') as f:
                data = pickle.load(f)
                model = data['model']
                scaler = data['scaler']
                feature_names = data['features']
                model_metrics = data.get('metrics', {})
                print(f"✅ Model loaded! Accuracy: {model_metrics.get('test_accuracy', 0):.2%}")
        except Exception as e:
            print(f"❌ Error: {e}")
            train_model()
    else:
        train_model()

load_model()

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    """Manual prediction endpoint"""
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        data = request.json
        
        input_data = pd.DataFrame([{
            'soil_height': float(data['soil_height']),
            'rainfall_intensity': float(data['rainfall_intensity']),
            'rainfall_duration': float(data['rainfall_duration']),
            'river_water_level': float(data['river_water_level']),
            'drainage_capability': float(data['drainage_capability']),
            'urbanization_index': float(data['urbanization_index']),
            'deforestation_index': float(data['deforestation_index']),
            'sea_level_rise': float(data['sea_level_rise']),
            'soil_moisture': float(data['soil_moisture']),
            'terrain_slope': float(data['terrain_slope']),
            'distance_to_river': float(data['distance_to_river']),
            'previous_24h_rainfall': float(data['previous_24h_rainfall']),
            'area_size': float(data.get('area_size', 50)),  # NEW with default
            'population_density': float(data.get('population_density', 5000))  # NEW with default
        }])
        
        input_data = input_data[feature_names]
        input_scaled = scaler.transform(input_data)
        prediction = int(model.predict(input_scaled)[0])
        probability = float(model.predict_proba(input_scaled)[0][1])
        
        # Calculate severity based on area size and population
        area_size = float(data.get('area_size', 50))
        pop_density = float(data.get('population_density', 5000))
        
        # Estimate people affected
        if prediction == 1:
            # Assume 30-70% of area could be flooded
            flood_coverage_pct = probability * 0.7
            affected_area = area_size * flood_coverage_pct
            estimated_people_affected = int(affected_area * pop_density)
        else:
            affected_area = 0
            estimated_people_affected = 0
        
        return jsonify({
            'flash_flood': prediction,
            'probability': probability,
            'model_type': 'Random Forest (BMKG-Enhanced)',
            'parameters_used': 14,
            'model_accuracy': model_metrics.get('test_accuracy', 0),
            'data_source': 'Manual Input',
            'severity_assessment': {
                'area_size_km2': area_size,
                'population_density': pop_density,
                'estimated_flood_coverage_km2': round(affected_area, 2),
                'estimated_people_affected': estimated_people_affected,
                'severity_level': 'High' if estimated_people_affected > 50000 else 'Moderate' if estimated_people_affected > 10000 else 'Low'
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict-from-bmkg', methods=['POST', 'OPTIONS'])
def predict_from_bmkg():
    """Get prediction using real BMKG data"""
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        data = request.json
        city_name = data.get('city_name', 'Cikarang')
        
        print(f"\n🌤️ Fetching BMKG data for: {city_name}")
        bmkg_data = fetch_bmkg_data(city_name)
        
        if not bmkg_data:
            return jsonify({'error': f'Failed to fetch BMKG data for {city_name}'}), 400
        
        params = estimate_flood_parameters_from_bmkg(bmkg_data, city_name)
        
        if not params:
            return jsonify({'error': 'Failed to process BMKG data'}), 500
        
        # Make prediction
        input_data = pd.DataFrame([{
            'soil_height': params['soil_height'],
            'rainfall_intensity': params['rainfall_intensity'],
            'rainfall_duration': params['rainfall_duration'],
            'river_water_level': params['river_water_level'],
            'drainage_capability': params['drainage_capability'],
            'urbanization_index': params['urbanization_index'],
            'deforestation_index': params['deforestation_index'],
            'sea_level_rise': params['sea_level_rise'],
            'soil_moisture': params['soil_moisture'],
            'terrain_slope': params['terrain_slope'],
            'distance_to_river': params['distance_to_river'],
            'previous_24h_rainfall': params['previous_24h_rainfall'],
            'area_size': params['area_size'],  # NEW
            'population_density': params['population_density']  # NEW
        }])
        
        input_data = input_data[feature_names]
        input_scaled = scaler.transform(input_data)
        prediction = int(model.predict(input_scaled)[0])
        probability = float(model.predict_proba(input_scaled)[0][1])
        
        # Calculate severity based on area and population
        area_size = params['area_size']
        pop_density = params['population_density']
        
        if prediction == 1:
            flood_coverage_pct = probability * 0.7
            affected_area = area_size * flood_coverage_pct
            estimated_people_affected = int(affected_area * pop_density)
        else:
            affected_area = 0
            estimated_people_affected = 0
        
        print(f"✅ Prediction complete: {probability:.1%} flood risk")
        print(f"   Area type: {params['area_type']} ({area_size} km²)")
        print(f"   Estimated affected: {estimated_people_affected:,} people")
        
        return jsonify({
            'flash_flood': prediction,
            'probability': probability,
            'location': params['location_name'],
            'area_info': {
                'area_type': params['area_type'],
                'area_size_km2': area_size,
                'population_density': pop_density,
                'total_population': int(area_size * pop_density)
            },
            'severity_assessment': {
                'estimated_flood_coverage_km2': round(affected_area, 2),
                'estimated_flood_coverage_pct': round(flood_coverage_pct * 100, 1),
                'estimated_people_affected': estimated_people_affected,
                'severity_level': 'Critical' if estimated_people_affected > 100000 else 'High' if estimated_people_affected > 50000 else 'Moderate' if estimated_people_affected > 10000 else 'Low'
            },
            'bmkg_data': {
                'temperature': params['bmkg_temperature'],
                'humidity': params['bmkg_humidity'],
                'weather': params['bmkg_weather'],
            },
            'estimated_parameters': {
                'rainfall_intensity': params['rainfall_intensity'],
                'rainfall_duration': params['rainfall_duration'],
                'soil_moisture': params['soil_moisture'],
                'river_water_level': params['river_water_level'],
                'previous_24h_rainfall': params['previous_24h_rainfall']
            },
            'model_type': 'Random Forest (BMKG-Enhanced)',
            'data_source': 'Real BMKG Weather Data',
            'model_accuracy': model_metrics.get('test_accuracy', 0)
        })
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/dataset-stats', methods=['GET'])
def dataset_stats():
    """Get dataset statistics for visualization"""
    try:
        df = generate_training_data(n_samples=1000)
        
        stats = {
            'total_samples': len(df),
            'flood_cases': int(df['flash_flood'].sum()),
            'no_flood_cases': int(len(df) - df['flash_flood'].sum())
        }
        
        return jsonify(stats)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/feature-importance', methods=['GET'])
def get_feature_importance():
    """Get model feature importance"""
    try:
        if model is None:
            return jsonify({'error': 'Model not trained'}), 400
        
        importance_data = []
        for feature, importance in zip(feature_names, model.feature_importances_):
            importance_data.append({
                'feature': feature,
                'importance': float(importance)
            })
        
        importance_data.sort(key=lambda x: x['importance'], reverse=True)
        
        return jsonify({
            'feature_importance': importance_data,
            'model_metrics': model_metrics
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'running',
        'model_loaded': model is not None,
        'parameters': len(feature_names) if feature_names else 0,
        'metrics': model_metrics,
        'bmkg_integration': True
    })

@app.route('/')
def home():
    """Serve HTML"""
    try:
        # Try to find index.html in current directory
        if os.path.exists('index.html'):
            return send_from_directory('.', 'index.html')
        else:
            # Return a simple debug page if index.html not found
            return '''
            <html>
            <body style="font-family: Arial; padding: 40px; background: #f0f0f0;">
                <h1 style="color: red;">⚠️ index.html not found!</h1>
                <p><strong>Current directory:</strong> {}</p>
                <p><strong>Files in directory:</strong></p>
                <ul>
                {}
                </ul>
                <hr>
                <h2>How to fix:</h2>
                <ol>
                    <li>Create a file named <code>index.html</code></li>
                    <li>Save it in the same folder as <code>app.py</code></li>
                    <li>Copy the HTML code from the artifact</li>
                    <li>Refresh this page</li>
                </ol>
            </body>
            </html>
            '''.format(
                os.getcwd(),
                ''.join([f'<li>{f}</li>' for f in os.listdir('.')])
            )
    except Exception as e:
        return f'<h1>Error: {str(e)}</h1>'

if __name__ == '__main__':
    print("="*70)
    print("FLASH FLOOD PREDICTION - BMKG INTEGRATION")
    print("AOL Artificial Intelligence - 3rd Semester")
    print("Kelompok 4")
    print("="*70)
    print("\n✨ Features:")
    print("  • Real BMKG Weather Data")
    print("  • City Name Input (Jakarta, Bandung, Surabaya, etc.)")
    print("  • 14 Parameters ML Model (NEW: Area Size + Population Density)")
    print("  • 8,000 Training Samples")
    print("  • Flood Severity Assessment")
    print("  • Estimated People Affected Calculation")
    print("\n🌐 Server starting at http://localhost:5000")
    print("\n⚠️  Keep this terminal open!")
    print("="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)