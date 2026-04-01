from flask import Flask, render_template, request, jsonify
from pymongo import MongoClient
from datetime import datetime
from collections import Counter

app = Flask(__name__)

# ------------------- MONGODB -------------------
MONGO_URI = "mongodb+srv://varshneysanidhya_db_user:IdlnLOhPaoW7XC6r@cluster0.1z1jtnl.mongodb.net/?appName=Cluster0"

client = MongoClient(MONGO_URI)
db = client['vehicledb']

collection = db['detectiondata']
blacklist_collection = db['blacklist']
alerts_collection = db['alerts']


# ------------------- DASHBOARD -------------------
@app.route('/', methods=['GET'])
def dashboard():

    start_date = request.args.get('start')
    end_date = request.args.get('end')
    vehicle_type = request.args.get('vehicleType')
    location = request.args.get('location')

    query = {}

    if start_date and end_date:
        query["timestamp"] = {
            "$gte": datetime.fromisoformat(start_date),
            "$lte": datetime.fromisoformat(end_date)
        }

    if vehicle_type:
        query["vehicleType"] = vehicle_type

    if location:
        query["location"] = location

    data = list(collection.find(query))

    total = len(data)


    if collection.count_documents({}) == 0:
        alerts_collection.delete_many({})

    vehicle_types = Counter([d.get("vehicleType", "Unknown") for d in data])
    colors = Counter([d.get("color", "Unknown") for d in data])
    brands = Counter([d.get("brandName", "Unknown") for d in data])
    locations = Counter([d.get("location", "Unknown") for d in data])

    hours = []
    for d in data:
        ts = d.get("timestamp")
        if ts:
            hours.append(ts.hour)
    hourly = Counter(hours)

    plates = Counter([d.get("number") for d in data])
    top_plates = plates.most_common(5)

    alerts = list(alerts_collection.find().sort("timestamp", -1).limit(5))

    return render_template(
        "dasboard.html",
        total=total,
        vehicle_types=vehicle_types,
        colors=colors,
        brands=brands,
        locations=locations,
        hourly=hourly,
        top_plates=top_plates,
        alerts=alerts
    )


# ------------------- REALTIME API -------------------
@app.route('/realtime-data')
def realtime_data():
    data = list(collection.find())
    total = len(data)

    # 🔥 FIX: Sync empty databases. If main data is cleared, wipe old alerts
    if total == 0:
        alerts_collection.delete_many({})

    # Calculate all stats for the live dashboard update
    vehicle_types = dict(Counter([d.get("vehicleType", "Unknown") for d in data]))
    colors = dict(Counter([d.get("color", "Unknown") for d in data]))
    brands = dict(Counter([d.get("brandName", "Unknown") for d in data]))
    
    hours = []
    for d in data:
        ts = d.get("timestamp")
        if ts:
            hours.append(ts.hour)
    hourly = dict(Counter(hours))
    
    plates = Counter([d.get("number") for d in data])
    top_plates = plates.most_common(5)

    latest_vehicle = collection.find_one(sort=[("_id", -1)])
    alert = None

    if latest_vehicle:
        latest_plate = latest_vehicle.get("number")
        black = blacklist_collection.find_one({"number": latest_plate})

        if black:
            last_alert = alerts_collection.find_one(sort=[("timestamp", -1)])
            
            
            if not last_alert or last_alert.get("number") != latest_plate:
                alert_reason = black.get("reason", "Blacklisted")
                alerts_collection.insert_one({
                    "number": latest_plate,
                    "reason": alert_reason,
                    "timestamp": datetime.now()
                })
                alert = {"number": latest_plate, "reason": alert_reason}
            

    # Fetch latest 5 alerts for the UI list and serialize for JSON
    alerts_list = list(alerts_collection.find().sort("timestamp", -1).limit(5))
    for a in alerts_list:
        a['_id'] = str(a['_id']) 
        a['timestamp'] = a['timestamp'].isoformat() if 'timestamp' in a else ""

    return jsonify({
        "total": total,
        "vehicle_types": vehicle_types,
        "colors": colors,
        "brands": brands,
        "hourly": hourly,
        "top_plates": top_plates,
        "alerts_list": alerts_list,
        "alert": alert
    })


# ------------------- RUN -------------------
if __name__ == "__main__":
    app.run(debug=True)