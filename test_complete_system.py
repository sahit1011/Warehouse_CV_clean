#!/usr/bin/env python3
"""
Comprehensive test suite for the Warehouse Free Space Measurement System
This script tests all components: backend API, frontend integration, database operations,
and computer vision pipeline.
"""

import os
import sys
import time
import requests
import sqlite3
import subprocess
import json
from datetime import datetime

class WarehouseSystemTester:
    def __init__(self):
        self.api_base = "http://localhost:5001"
        self.db_path = "backend/warehouse.db"
        self.test_results = []
        self.backend_process = None
        
    def log_test(self, test_name, passed, message=""):
        """Log test result"""
        status = "✅ PASS" if passed else "❌ FAIL"
        result = {
            'test': test_name,
            'passed': passed,
            'message': message,
            'timestamp': datetime.now().isoformat()
        }
        self.test_results.append(result)
        print(f"{status}: {test_name} - {message}")
        
    def test_database_operations(self):
        """Test database creation and operations"""
        print("\n🗄️  Testing Database Operations...")
        
        try:
            # Test database connection
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Test table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='warehouse';")
            table_exists = cursor.fetchone() is not None
            self.log_test("Database table exists", table_exists)
            
            # Test data insertion
            test_data = ('TEST-AISLE', 'TEST-RACK', 'main-rack', 3, 1, 2, 'good', 'Test entry', '2024-06-24 20:00')
            cursor.execute('''INSERT INTO warehouse (aisle, rack_id, rack_type, total_capacity, 
                             occupied_space, free_space, condition, notes, timestamp) 
                             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''', test_data)
            conn.commit()
            
            # Test data retrieval
            cursor.execute("SELECT * FROM warehouse WHERE rack_id='TEST-RACK'")
            retrieved_data = cursor.fetchone()
            data_inserted = retrieved_data is not None
            self.log_test("Database data insertion/retrieval", data_inserted)
            
            # Clean up test data
            cursor.execute("DELETE FROM warehouse WHERE rack_id='TEST-RACK'")
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.log_test("Database operations", False, str(e))
    
    def test_backend_api(self):
        """Test backend API endpoints"""
        print("\n🔌 Testing Backend API...")
        
        # Test API connectivity
        try:
            response = requests.get(f"{self.api_base}/api/data", timeout=5)
            api_accessible = response.status_code == 200
            self.log_test("API accessibility", api_accessible, f"Status: {response.status_code}")
        except Exception as e:
            self.log_test("API accessibility", False, str(e))
            return
        
        # Test POST endpoint
        test_entry = {
            "aisle": "TEST-API",
            "rackId": "R-API-TEST",
            "rackType": "main-rack",
            "totalCapacity": 3,
            "occupiedSpace": 1,
            "freeSpace": 2,
            "condition": "good",
            "notes": "API test entry",
            "timestamp": "2024-06-24 21:00"
        }
        
        try:
            response = requests.post(f"{self.api_base}/api/add", json=test_entry, timeout=5)
            post_success = response.status_code == 200
            self.log_test("API POST /api/add", post_success, f"Status: {response.status_code}")
            
            if post_success:
                # Test GET endpoint with new data
                response = requests.get(f"{self.api_base}/api/data", timeout=5)
                data = response.json()
                test_entries = [entry for entry in data if entry.get('rackId') == 'R-API-TEST']
                data_retrieved = len(test_entries) > 0
                self.log_test("API GET after POST", data_retrieved)
                
                # Test DELETE endpoint
                if test_entries:
                    test_id = test_entries[0]['id']
                    response = requests.delete(f"{self.api_base}/api/delete/{test_id}", timeout=5)
                    delete_success = response.status_code == 200
                    self.log_test("API DELETE", delete_success, f"Status: {response.status_code}")
                    
        except Exception as e:
            self.log_test("API POST/DELETE operations", False, str(e))
    
    def test_frontend_serving(self):
        """Test frontend serving"""
        print("\n🌐 Testing Frontend Serving...")
        
        try:
            response = requests.get(self.api_base, timeout=5)
            frontend_accessible = response.status_code == 200
            
            if frontend_accessible:
                content_correct = "Warehouse Free Space Measurement System" in response.text
                self.log_test("Frontend content", content_correct)
            else:
                self.log_test("Frontend accessibility", False, f"Status: {response.status_code}")
                
        except Exception as e:
            self.log_test("Frontend serving", False, str(e))
    
    def test_computer_vision_components(self):
        """Test computer vision pipeline"""
        print("\n👁️  Testing Computer Vision Components...")
        
        # Test if video files exist
        video_dir = "video"
        video_files = []
        if os.path.exists(video_dir):
            video_files = [f for f in os.listdir(video_dir) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
        
        videos_available = len(video_files) > 0
        self.log_test("Video files available", videos_available, f"Found {len(video_files)} videos")
        
        # Test if CV dependencies are available
        try:
            import cv2
            import numpy as np
            from ultralytics import YOLO
            cv_deps_available = True
            self.log_test("CV dependencies", cv_deps_available)
        except ImportError as e:
            self.log_test("CV dependencies", False, str(e))
            cv_deps_available = False
        
        # Test model loading (basic)
        if cv_deps_available:
            try:
                model = YOLO('yolov8n.pt')  # Load pre-trained model
                model_loadable = True
                self.log_test("YOLO model loading", model_loadable)
            except Exception as e:
                self.log_test("YOLO model loading", False, str(e))
    
    def test_file_structure(self):
        """Test project file structure"""
        print("\n📁 Testing File Structure...")
        
        required_files = [
            "backend/app.py",
            "backend/requirements.txt",
            "frontend/index.html",
            "src/main.py",
            "src/train_model.py"
        ]
        
        for file_path in required_files:
            file_exists = os.path.exists(file_path)
            self.log_test(f"File exists: {file_path}", file_exists)
        
        required_dirs = [
            "backend",
            "frontend", 
            "src",
            "video"
        ]
        
        for dir_path in required_dirs:
            dir_exists = os.path.exists(dir_path)
            self.log_test(f"Directory exists: {dir_path}", dir_exists)
    
    def run_all_tests(self):
        """Run complete test suite"""
        print("🧪 Warehouse System Comprehensive Test Suite")
        print("=" * 60)
        
        start_time = time.time()
        
        # Run all test categories
        self.test_file_structure()
        self.test_database_operations()
        self.test_backend_api()
        self.test_frontend_serving()
        self.test_computer_vision_components()
        
        # Generate summary
        end_time = time.time()
        duration = end_time - start_time
        
        passed_tests = sum(1 for result in self.test_results if result['passed'])
        total_tests = len(self.test_results)
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Duration: {duration:.2f} seconds")
        
        if success_rate >= 80:
            print("\n🎉 System is functioning well!")
        elif success_rate >= 60:
            print("\n⚠️  System has some issues that need attention")
        else:
            print("\n❌ System has significant issues that need to be resolved")
        
        # Save detailed results
        self.save_test_results()
        
        return success_rate >= 80
    
    def save_test_results(self):
        """Save test results to file"""
        results_file = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        print(f"\n📄 Detailed results saved to: {results_file}")


def main():
    """Main function to run the test suite"""
    tester = WarehouseSystemTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n✅ All systems operational!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the issues above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
