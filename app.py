from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import os
import requests

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

scaler = None
model = None
feature_names = None
model_metrics = {}

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
}

BMKG_API_URL = "https://api.bmkg.go.id/publik/prakiraan-cuaca"


def get_city_code(city_name):
    return CITY_CODES.get(city_name.lower().strip(), CITY_CODES['cikarang'])


def fetch_bmkg_data(city_name):
    """
    SAFE BMKG FETCH:
    - pakai headers
    - handle data kosong
    - fallback ke simulated data
    """
    adm4 = get_city_code(city_name)
    url = f"{BMKG_API_URL}?adm4={adm4}"

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json"
    }

    try:
        print(f"🌐 Fetch BMKG: {city_name} ({adm4})")
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        data = response.json()
        cuaca = data.get("data", [])

        if not cuaca or not cuaca[0].get("cuaca"):
            raise ValueError("BMKG cuaca kosong")

        weather = cuaca[0]["cuaca"][0]
        lokasi = data.get("lokasi", {})

        return {
            "weather": {
                "temperature": weather.get("t", 28),
                "humidity": weather.get("hu", 75),
                "weather_desc": weather.get("weather_desc", "Cerah"),
                "total_cloud_cover": weather.get("tcc", 70),
                "rainfall_prob": weather.get("tp", 40),
            },
            "location": {
                "desa": lokasi.get("desa", city_name),
                "kecamatan": lokasi.get("kecamatan", city_name),
                "kotkab": lokasi.get("kotkab", city_name),
            }
        }

    except Exception as e:
        print(f"⚠️ BMKG FAILED → DEMO MODE ({e})")

        return {
            "weather": {
                "temperature": 28,
                "humidity": 78,
                "weather_desc": "Hujan Ringan",
                "total_cloud_cover": 80,
                "rainfall_prob": 60,
            },
            "location": {
                "desa": city_name,
                "kecamatan": city_name,
                "kotkab": city_name,
            }
        }


def estimate_flood_parameters_from_bmkg(bmkg, city):
    w = bmkg["weather"]

    rainfall_intensity = (
        np.random.uniform(25, 50) if w["rainfall_prob"] > 60 else
        np.random.uniform(10, 25)
    )

    return {
        "soil_height": 15,
        "rainfall_intensity": round(rainfall_intensity, 1),
        "rainfall_duration": round(w["rainfall_prob"] / 100 * 8, 1),
        "river_water_level": round(w["rainfall_prob"] / 100 * 6, 1),
        "drainage_capability": 55,
        "urbanization_index": 70,
        "deforestation_index": 50,
        "sea_level_rise": 0.3,
        "soil_moisture": min(w["humidity"], 100),
        "terrain_slope": 2,
        "distance_to_river": 1.0,
        "previous_24h_rainfall": round(rainfall_intensity * 2, 1),
        "area_size": 85,
        "population_density": 8500,
        "area_type": "District",
        "bmkg_temperature": w["temperature"],
        "bmkg_humidity": w["humidity"],
        "bmkg_weather": w["weather_desc"],
        "location_name": f"{city}"
    }


def generate_training_data(n=8000):
    np.random.seed(42)
    df = pd.DataFrame({
        "soil_height": np.random.normal(30, 40, n),
        "rainfall_intensity": np.random.gamma(2, 12, n),
        "rainfall_duration": np.random.gamma(2, 3, n),
        "previous_24h_rainfall": np.random.gamma(2.5, 20, n),
        "river_water_level": np.random.gamma(2, 1.5, n),
        "distance_to_river": np.random.exponential(2, n),
        "drainage_capability": np.random.beta(3, 2.5, n) * 100,
        "urbanization_index": np.random.beta(2, 2, n) * 100,
        "deforestation_index": np.random.beta(2, 2.5, n) * 100,
        "sea_level_rise": np.random.gamma(1.5, 0.25, n),
        "soil_moisture": np.random.beta(4, 2, n) * 100,
        "terrain_slope": np.random.gamma(2, 5, n),
        "area_size": np.random.gamma(3, 15, n),
        "population_density": np.random.gamma(2, 2000, n),
    })

    risk = (
        df["rainfall_intensity"] * 0.2 +
        df["soil_moisture"] * 0.1 +
        (100 - df["drainage_capability"]) * 0.2
    )

    df["flash_flood"] = (risk > np.percentile(risk, 68)).astype(int)
    return df


def load_model():
    global model, scaler, feature_names, model_metrics

    if os.path.exists("flood_model_bmkg.pkl"):
        with open("flood_model_bmkg.pkl", "rb") as f:
            data = pickle.load(f)
            model = data["model"]
            scaler = data["scaler"]
            feature_names = data["features"]
            model_metrics = data["metrics"]
    else:
        df = generate_training_data()
        X = df.drop("flash_flood", axis=1)
        y = df["flash_flood"]
        feature_names = X.columns.tolist()

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        model = RandomForestClassifier(n_estimators=250, random_state=42)
        model.fit(Xs, y)

        model_metrics = {"accuracy": model.score(Xs, y)}

        with open("flood_model_bmkg.pkl", "wb") as f:
            pickle.dump({
                "model": model,
                "scaler": scaler,
                "features": feature_names,
                "metrics": model_metrics
            }, f)


load_model()


@app.route("/predict-from-bmkg", methods=["POST"])
def predict_from_bmkg():
    city = request.json.get("city_name", "Cikarang")

    bmkg = fetch_bmkg_data(city)
    params = estimate_flood_parameters_from_bmkg(bmkg, city)

    X = pd.DataFrame([{k: params[k] for k in feature_names}])
    Xs = scaler.transform(X)

    prob = float(model.predict_proba(Xs)[0][1])
    pred = int(prob > 0.5)

    affected_area = params["area_size"] * prob * 0.7
    people = int(affected_area * params["population_density"])

    return jsonify({
    "flash_flood": pred,
    "probability": prob,
    "location": params["location_name"],
    "area_info": {
        "area_type": params["area_type"],
        "area_size_km2": params["area_size"],
        "population_density": params["population_density"],
        "total_population": int(params["area_size"] * params["population_density"])
    },
    "severity_assessment": {
        "estimated_flood_coverage_km2": round(affected_area, 2),
        "estimated_people_affected": people,
        "severity_level": "High" if people > 50000 else "Moderate"
    },

    "estimated_parameters": {
        "rainfall_intensity": params["rainfall_intensity"],
        "rainfall_duration": params["rainfall_duration"],
        "soil_moisture": params["soil_moisture"],
        "river_water_level": params["river_water_level"],
        "previous_24h_rainfall": params["previous_24h_rainfall"]
    },

    "bmkg_data": {
        "temperature": params["bmkg_temperature"],
        "humidity": params["bmkg_humidity"],
        "weather": params["bmkg_weather"]
    },
    "model_type": "Random Forest (BMKG-Enhanced)",
    "data_source": "BMKG (Real / Simulated)",
    "model_accuracy": model_metrics.get("accuracy", 0)
})

@app.route("/predict", methods=["POST"])
def predict_manual():
    try:
        data = request.json

        input_data = pd.DataFrame([{
            "soil_height": float(data["soil_height"]),
            "rainfall_intensity": float(data["rainfall_intensity"]),
            "rainfall_duration": float(data["rainfall_duration"]),
            "river_water_level": float(data["river_water_level"]),
            "drainage_capability": float(data["drainage_capability"]),
            "urbanization_index": float(data["urbanization_index"]),
            "deforestation_index": float(data["deforestation_index"]),
            "sea_level_rise": float(data["sea_level_rise"]),
            "soil_moisture": float(data["soil_moisture"]),
            "terrain_slope": float(data["terrain_slope"]),
            "distance_to_river": float(data["distance_to_river"]),
            "previous_24h_rainfall": float(data["previous_24h_rainfall"]),
            "area_size": float(data.get("area_size", 50)),
            "population_density": float(data.get("population_density", 5000)),
        }])

        input_data = input_data[feature_names]
        input_scaled = scaler.transform(input_data)

        prob = float(model.predict_proba(input_scaled)[0][1])
        pred = int(prob > 0.5)

        area = float(data.get("area_size", 50))
        pop = float(data.get("population_density", 5000))

        affected_area = area * prob * 0.7 if pred == 1 else 0
        people = int(affected_area * pop)

        return jsonify({
            "flash_flood": pred,
            "probability": prob,
            "model_type": "Random Forest (Manual Input)",
            "data_source": "Manual Input",
            "model_accuracy": model_metrics.get("accuracy", 0),

            "severity_assessment": {
                "area_size_km2": area,
                "population_density": pop,
                "estimated_flood_coverage_km2": round(affected_area, 2),
                "estimated_people_affected": people,
                "severity_level": (
                    "High" if people > 50000 else
                    "Moderate" if people > 10000 else
                    "Low"
                )
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({"status": "running"})


@app.route("/")
def home():
    return send_from_directory(".", "index.html")


if __name__ == "__main__":
    print("🌊 Flash Flood Prediction Server Running")
    app.run(debug=True, port=5000)
