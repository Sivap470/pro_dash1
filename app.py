from flask import Flask, render_template
from tensorflow.keras.models import load_model
import numpy as np
import requests
import joblib
import pickle
import datetime
from twilio.rest import Client
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

app = Flask(__name__)

# Load ML model, scaler, and encoder
model = load_model("bilstm_water_quality_model.h5")
scaler = joblib.load("scaler.save")
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Load Twilio credentials from environment variables
TWILIO_SID = os.getenv("TWILIO_SID")

TWILIO_TOKEN = os.getenv("TWILIO_TOKEN")
TWILIO_FROM = os.getenv("TWILIO_FROM")
TWILIO_TO = os.getenv("TWILIO_TO")

# ThingSpeak Configuration
CHANNEL_ID = "2972863"
READ_API_KEY = "MZY58DY1BLJ3JTGRY"
PH_FIELD = 1
EC_FIELD = 2

# Real-time buffers
ph_history = []
ec_history = []
time_history = []
MAX_HISTORY = 10

# Alert flags
alert_sent = {
    "ph": False,
    "ec": False
}

def send_alert(parameter, value):
    try:
        client = Client(TWILIO_SID, TWILIO_TOKEN)
        msg = client.messages.create(
            body=f"⚠️ ALERT: {parameter.upper()} abnormal. Value = {value}",
            from_=TWILIO_FROM,
            to=TWILIO_TO
        )
        print(f"Alert sent: {msg.sid}")
    except Exception as e:
        print("Twilio error:", e)

def fetch_live_data():
    url = f"https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds/last.json?api_key={READ_API_KEY}"
    try:
        response = requests.get(url, timeout=5)
        data = response.json()
        ph = float(data[f'field{PH_FIELD}'])
        ec = float(data[f'field{EC_FIELD}'])

        now = datetime.datetime.now().strftime("%H:%M:%S")
        ph_history.append(ph)
        ec_history.append(ec)
        time_history.append(now)

        if len(ph_history) > MAX_HISTORY:
            ph_history.pop(0)
            ec_history.pop(0)
            time_history.pop(0)

        return ph, ec
    except:
        return None, None

def predict_quality(ph, ec):
    scaled = scaler.transform([[ph, ec]])
    reshaped = scaled.reshape((1, 1, 2))
    pred = model.predict(reshaped)
    label_idx = np.argmax(pred, axis=1)[0]
    return le.inverse_transform([label_idx])[0]

@app.route("/")
def index():
    ph, ec = fetch_live_data()

    if ph is not None and ec is not None:
        label = predict_quality(ph, ec)
        sensor_status = "online"

        # Alert check
        if ph < 6.5 or ph > 8.5:
            if not alert_sent["ph"]:
                send_alert("pH", ph)
                alert_sent["ph"] = True
        else:
            alert_sent["ph"] = False

        if ec < 50 or ec > 1500:
            if not alert_sent["ec"]:
                send_alert("EC", ec)
                alert_sent["ec"] = True
        else:
            alert_sent["ec"] = False
    else:
        label = "Unavailable"
        ph, ec = "N/A", "N/A"
        sensor_status = "offline"

    return render_template(
        "index.html",
        ph=ph,
        ec=ec,
        label=label,
        sensor_status=sensor_status,
        time_labels=time_history,
        ph_values=ph_history,
        ec_values=ec_history
    )

if __name__ == "__main__":
    app.run(debug=True)
