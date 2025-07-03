# app/app.py
# This script will run the Flask web server for our dashboard.

from flask import Flask, jsonify, render_template

app = Flask(__name__)

@app.route('/')
def dashboard():
    """
    Renders the main dashboard page.
    """
    return render_template('dashboard.html')

@app.route('/api/data')
def get_data():
    """
    API endpoint to provide warehouse data to the dashboard.
    """
    # This will be replaced with a call to our database or data source
    dummy_data = {
        "aisle_1": {"total": 100, "occupied": 75, "free": 25},
        "aisle_2": {"total": 100, "occupied": 50, "free": 50},
        "aisle_3": {"total": 120, "occupied": 110, "free": 10},
    }
    return jsonify(dummy_data)

if __name__ == '__main__':
    app.run(debug=True) 