import os
import sqlite3
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import pandas as pd

DB_PATH = os.path.join(os.path.dirname(__file__), 'warehouse.db')
FRONTEND_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../frontend'))

app = Flask(__name__, static_folder=os.path.join(FRONTEND_PATH, 'static'))
CORS(app)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS warehouse (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        aisle TEXT,
        rack_id TEXT,
        rack_type TEXT,
        total_capacity INTEGER,
        occupied_space INTEGER,
        free_space INTEGER,
        condition TEXT,
        notes TEXT,
        timestamp TEXT
    )''')
    conn.commit()
    conn.close()

@app.route('/')
def serve_index():
    return send_from_directory(FRONTEND_PATH, 'index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory(os.path.join(FRONTEND_PATH, 'static'), path)

@app.route('/api/data', methods=['GET'])
def get_data():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT * FROM warehouse')
    rows = c.fetchall()
    conn.close()
    # Map DB fields to camelCase keys for frontend compatibility
    columns = ['id', 'aisle', 'rack_id', 'rack_type', 'total_capacity', 'occupied_space', 'free_space', 'condition', 'notes', 'timestamp']
    data = [dict(zip(columns, row)) for row in rows]
    camel_data = []
    for d in data:
        camel_data.append({
            'id': d['id'],
            'aisle': d['aisle'],
            'rackId': d['rack_id'],
            'rackType': d['rack_type'],
            'totalCapacity': d['total_capacity'],
            'occupiedSpace': d['occupied_space'],
            'freeSpace': d['free_space'],
            'condition': d['condition'],
            'notes': d['notes'],
            'timestamp': d['timestamp']
        })
    return jsonify(camel_data)

@app.route('/api/add', methods=['POST'])
def add_entry():
    entry = request.get_json(force=True)
    required_fields = ['aisle', 'rackId', 'rackType', 'totalCapacity', 'occupiedSpace', 'freeSpace', 'condition', 'notes', 'timestamp']
    if not all(field in entry for field in required_fields):
        return jsonify({'status': 'error', 'message': 'Missing fields in request'}), 400
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''INSERT INTO warehouse (aisle, rack_id, rack_type, total_capacity, occupied_space, free_space, condition, notes, timestamp)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
              (entry['aisle'], entry['rackId'], entry['rackType'], entry['totalCapacity'], entry['occupiedSpace'], entry['freeSpace'], entry['condition'], entry['notes'], entry['timestamp']))
    conn.commit()
    conn.close()
    return jsonify({'status': 'success'})

@app.route('/api/delete/<int:entry_id>', methods=['DELETE'])
def delete_entry(entry_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('DELETE FROM warehouse WHERE id=?', (entry_id,))
    conn.commit()
    conn.close()
    return jsonify({'status': 'deleted'})

@app.route('/api/export', methods=['GET'])
def export_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query('SELECT * FROM warehouse', conn)
    conn.close()
    export_path = os.path.join(os.path.dirname(__file__), 'warehouse_export.xlsx')
    df.to_excel(export_path, index=False)
    return send_file(export_path, as_attachment=True)

if __name__ == '__main__':
    init_db()
    app.run(port=5001, debug=True) 