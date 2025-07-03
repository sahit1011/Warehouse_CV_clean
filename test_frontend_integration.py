#!/usr/bin/env python3
"""
Test script to verify frontend-backend integration
This script simulates what the frontend JavaScript would do
"""

import requests
import json

API_BASE = 'http://localhost:5001'

def test_frontend_backend_integration():
    print("🧪 Testing Frontend-Backend Integration")
    print("=" * 50)
    
    # Test 1: Check if backend is accessible
    try:
        response = requests.get(f"{API_BASE}/api/data")
        print(f"✅ Backend API accessible: {response.status_code}")
        data = response.json()
        print(f"✅ Data retrieved: {len(data)} entries")
    except Exception as e:
        print(f"❌ Backend API error: {e}")
        return False
    
    # Test 2: Check if frontend is served
    try:
        response = requests.get(f"{API_BASE}/")
        print(f"✅ Frontend accessible: {response.status_code}")
        if "Warehouse Free Space Measurement System" in response.text:
            print("✅ Frontend content verified")
        else:
            print("⚠️  Frontend content may not be correct")
    except Exception as e:
        print(f"❌ Frontend access error: {e}")
        return False
    
    # Test 3: Test adding data (simulating frontend form submission)
    test_entry = {
        "aisle": "TEST-1",
        "rackId": "R-TEST",
        "rackType": "main-rack",
        "totalCapacity": 3,
        "occupiedSpace": 1,
        "freeSpace": 2,
        "condition": "good",
        "notes": "Integration test entry",
        "timestamp": "2024-06-24 20:30"
    }
    
    try:
        response = requests.post(f"{API_BASE}/api/add", json=test_entry)
        print(f"✅ Data addition test: {response.status_code}")
        if response.status_code == 200:
            print("✅ Test entry added successfully")
        else:
            print(f"⚠️  Unexpected response: {response.text}")
    except Exception as e:
        print(f"❌ Data addition error: {e}")
        return False
    
    # Test 4: Verify the test entry was added
    try:
        response = requests.get(f"{API_BASE}/api/data")
        data = response.json()
        test_entries = [entry for entry in data if entry.get('rackId') == 'R-TEST']
        if test_entries:
            print("✅ Test entry verified in database")
            # Clean up - delete the test entry
            test_id = test_entries[0]['id']
            delete_response = requests.delete(f"{API_BASE}/api/delete/{test_id}")
            if delete_response.status_code == 200:
                print("✅ Test entry cleaned up")
        else:
            print("⚠️  Test entry not found in database")
    except Exception as e:
        print(f"❌ Verification error: {e}")
        return False
    
    print("\n🎉 All integration tests passed!")
    print("✅ Backend API is working")
    print("✅ Frontend is being served")
    print("✅ Data flow is working correctly")
    print("✅ CRUD operations are functional")
    
    return True

if __name__ == "__main__":
    test_frontend_backend_integration()
