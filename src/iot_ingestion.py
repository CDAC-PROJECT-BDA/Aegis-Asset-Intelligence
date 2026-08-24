import sqlite3
import pandas as pd
import time
import os

DB_PATH = 'iot_telemetry.db'
CSV_PATH = 'data/sensor.csv'

def setup_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS telemetry (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            sensor_00 REAL, sensor_01 REAL, sensor_02 REAL, sensor_03 REAL, sensor_04 REAL, sensor_05 REAL,
            sensor_06 REAL, sensor_07 REAL, sensor_08 REAL, sensor_09 REAL, sensor_10 REAL, sensor_11 REAL,
            sensor_12 REAL, sensor_13 REAL, sensor_14 REAL, sensor_16 REAL, sensor_17 REAL, sensor_18 REAL,
            sensor_19 REAL, sensor_20 REAL, sensor_21 REAL, sensor_22 REAL, sensor_23 REAL, sensor_24 REAL,
            sensor_25 REAL, sensor_26 REAL, sensor_27 REAL, sensor_28 REAL, sensor_29 REAL, sensor_30 REAL,
            sensor_31 REAL, sensor_32 REAL, sensor_33 REAL, sensor_34 REAL, sensor_35 REAL, sensor_36 REAL,
            sensor_37 REAL, sensor_38 REAL, sensor_39 REAL, sensor_40 REAL, sensor_41 REAL, sensor_42 REAL,
            sensor_43 REAL, sensor_44 REAL, sensor_45 REAL, sensor_46 REAL, sensor_47 REAL, sensor_48 REAL,
            sensor_49 REAL, sensor_50 REAL, sensor_51 REAL,
            machine_status TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS commands (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            command TEXT,
            handled INTEGER DEFAULT 0
        )
    ''')
    
    cursor.execute('DELETE FROM telemetry')
    conn.commit()
    conn.close()

def run_ingestion():
    setup_db()
    print(f"Loading {CSV_PATH} for playback...")
    if not os.path.exists(CSV_PATH):
        print(f"File {CSV_PATH} not found. Cannot run ingestion.")
        return
        
    df = pd.read_csv(CSV_PATH)
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    
    normal_df = df[df['machine_status'] == 'NORMAL'].reset_index(drop=True)
    broken_df = df[df['machine_status'] == 'BROKEN'].reset_index(drop=True)
    
    cols = [c for c in normal_df.columns if c not in ['timestamp', 'sensor_15']]
    
    conn = sqlite3.connect(DB_PATH)
    
    print("Starting IoT sensor ingestion loop...")
    normal_idx = 0
    broken_idx = 0
    
    while True:
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT id, command FROM commands WHERE handled = 0 ORDER BY id LIMIT 1")
            cmd_row = cursor.fetchone()
            
            row_to_insert = None
            if cmd_row and cmd_row[1] == 'INJECT_FAILURE':
                row_to_insert = broken_df.iloc[broken_idx % len(broken_df)].to_dict()
                broken_idx += 1
                cursor.execute("UPDATE commands SET handled = 1 WHERE id = ?", (cmd_row[0],))
                conn.commit()
                print("Injected FAILURE row from dataset.")
            else:
                row_to_insert = normal_df.iloc[normal_idx % len(normal_df)].to_dict()
                normal_idx += 1
            
            cols_to_insert = [c for c in cols if c in row_to_insert]
            placeholders = ', '.join(['?'] * len(cols_to_insert))
            col_names = ', '.join(cols_to_insert)
            
            values = [row_to_insert[c] for c in cols_to_insert]
            
            cursor.execute(f"INSERT INTO telemetry ({col_names}) VALUES ({placeholders})", values)
            conn.commit()
            
            cursor.execute("DELETE FROM telemetry WHERE id NOT IN (SELECT id FROM telemetry ORDER BY id DESC LIMIT 100)")
            conn.commit()
            
            time.sleep(1.5) 

        except Exception as e:
            print(f"Ingestion error: {e}")
            time.sleep(2)

if __name__ == "__main__":
    run_ingestion()
