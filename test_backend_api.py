import requests

API_BASE = 'http://127.0.0.1:5001'

# 1. Test POST /api/add
entry = {
    "aisle": "A-1",
    "rackId": "R-001",
    "rackType": "main-rack",
    "totalCapacity": 3,
    "occupiedSpace": 2,
    "freeSpace": 1,
    "condition": "good",
    "notes": "Test entry",
    "timestamp": "2024-06-24 20:00"
}
print("Testing POST /api/add ...")
r = requests.post(f"{API_BASE}/api/add", json=entry)
print("Status:", r.status_code, "Response:", r.json())

# 2. Test GET /api/data
print("\nTesting GET /api/data ...")
r = requests.get(f"{API_BASE}/api/data")
print("Status:", r.status_code)
print("Data:", r.json())

# 3. Test DELETE /api/delete/1 (delete the first entry if exists)
data = r.json()
if data:
    first_id = data[0]['id']
    print(f"\nTesting DELETE /api/delete/{first_id} ...")
    r = requests.delete(f"{API_BASE}/api/delete/{first_id}")
    print("Status:", r.status_code, "Response:", r.json())
else:
    print("\nNo data to delete.")

# 4. Test GET /api/data again
print("\nTesting GET /api/data after delete ...")
r = requests.get(f"{API_BASE}/api/data")
print("Status:", r.status_code)
print("Data:", r.json()) 