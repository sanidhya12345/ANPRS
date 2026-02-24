from pymongo import MongoClient
from datetime import datetime
MONGO_URI="mongodb+srv://varshneysanidhya_db_user:IdlnLOhPaoW7XC6r@cluster0.1z1jtnl.mongodb.net/?appName=Cluster0"

client=MongoClient(MONGO_URI)
print("Mongodb client is successfully connected")
db=client['vehicledb']
collection=db['detectiondata']
def insert_into_db(plate_number, color, vehicle_type, brand):

    vehicle_data = {
        "number": plate_number,
        "color": color,
        "vehicleType": vehicle_type,
        "brandName": brand,
        "location": "Gate-1",
        "timestamp": datetime.now()
    }

    collection.insert_one(vehicle_data)
    print("Inserted Successfully")

plate_number="UPAS12345"
color="green"
vehicleType="electric"
brandname="ford"
insert_into_db(plate_number,color,vehicleType,brandname)