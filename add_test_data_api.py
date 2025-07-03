import requests

API_BASE = 'http://127.0.0.1:5001'

entries = [
    {
        "aisle": "B-2",
        "rackId": "R-002",
        "rackType": "main-rack",
        "totalCapacity": 3,
        "occupiedSpace": 1,
        "freeSpace": 2,
        "condition": "good",
        "notes": "API entry 1",
        "timestamp": "2024-06-25 10:00"
    },
    {
        "aisle": "C-1",
        "rackId": "S-003",
        "rackType": "small-shelf",
        "totalCapacity": 1,
        "occupiedSpace": 0,
        "freeSpace": 1,
        "condition": "partially-blocked",
        "notes": "API entry 2",
        "timestamp": "2024-06-25 10:05"
    },
    {
        "aisle": "D-4",
        "rackId": "R-010",
        "rackType": "main-rack",
        "totalCapacity": 3,
        "occupiedSpace": 3,
        "freeSpace": 0,
        "condition": "damaged",
        "notes": "API entry 3",
        "timestamp": "2024-06-25 10:10"
    }
]

for entry in entries:
    r = requests.post(f"{API_BASE}/api/add", json=entry)
    print("Added:", entry["rackId"], "Status:", r.status_code, r.json()) 