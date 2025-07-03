import sqlite3

DB_PATH = 'backend/warehouse.db'

conn = sqlite3.connect(DB_PATH)
c = conn.cursor()

# List all tables
print('Tables:')
c.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = c.fetchall()
print(tables)

# Print first 5 rows from warehouse table if it exists
if any('warehouse' in t for t in tables):
    print('\nFirst 5 rows from warehouse table:')
    for row in c.execute('SELECT * FROM warehouse LIMIT 5;'):
        print(row)
else:
    print('\nNo warehouse table found.')

conn.close() 