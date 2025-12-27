"""
NATGRID Event Logs Data Generator
Generates event_logs.csv with 350+ events including anomalies
Enhanced with better anomaly labeling and more realistic patterns
"""

import pandas as pd
import random
from datetime import datetime, timedelta
import numpy as np


def generate_event_logs():
    random.seed(42)
    np.random.seed(42)

    events = []
    event_id = 1

    # User profiles for normal behavior
    normal_users = {
        'USR_1001': {'name': 'Rajesh Kumar', 'level': 'Level-2', 'location': 'Mumbai', 'shift': 'Day', 'department': 'Intelligence'},
        'USR_1002': {'name': 'Priya Sharma', 'level': 'Level-3', 'location': 'Delhi', 'shift': 'Day', 'department': 'Analysis'},
        'USR_1003': {'name': 'Amit Patel', 'level': 'Level-2', 'location': 'Bangalore', 'shift': 'Day', 'department': 'Cyber'},
        'USR_1004': {'name': 'Sneha Reddy', 'level': 'Level-1', 'location': 'Chennai', 'shift': 'Day', 'department': 'Operations'},
        'USR_1005': {'name': 'Vikram Singh', 'level': 'Level-3', 'location': 'Hyderabad', 'shift': 'Night', 'department': 'Security'},
        'USR_1006': {'name': 'Anjali Verma', 'level': 'Level-2', 'location': 'Pune', 'shift': 'Day', 'department': 'Intelligence'},
        'USR_1007': {'name': 'Rohit Mehta', 'level': 'Level-1', 'location': 'Kolkata', 'shift': 'Day', 'department': 'Support'},
        'USR_1008': {'name': 'Kavita Joshi', 'level': 'Level-2', 'location': 'Ahmedabad', 'shift': 'Day', 'department': 'Analysis'},
    }

    event_types = ['database_access', 'file_access', 'login', 'api_call', 'system_query', 'report_generation', 'data_export']
    devices = [f'DEV_A{str(i).zfill(2)}' for i in range(1, 21)]
    
    location_ips = {
        'Mumbai': '192.168.1', 'Delhi': '192.168.2', 'Bangalore': '192.168.3',
        'Chennai': '192.168.4', 'Hyderabad': '192.168.5', 'Pune': '192.168.6',
        'Kolkata': '192.168.7', 'Ahmedabad': '192.168.8'
    }

    start_date = datetime(2024, 11, 1)

    # ========== GENERATE NORMAL EVENTS (298 rows - 85%) ==========
    for i in range(298):
        user_id = random.choice(list(normal_users.keys()))
        user_profile = normal_users[user_id]
        
        # Normal working hours
        if user_profile['shift'] == 'Day':
            hour = random.randint(9, 18)
        else:
            # Night shift: 22-23 or 0-6
            hour = random.choice(list(range(22, 24)) + list(range(0, 7)))
        
        timestamp = start_date + timedelta(days=random.randint(0, 45), hours=hour, minutes=random.randint(0, 59))
        
        event_type = random.choice(event_types)
        
        # Normal parameters based on access level
        if user_profile['level'] == 'Level-1':
            access_level = 'Level-1'
            transaction_amount = None if random.random() > 0.3 else random.randint(1000, 50000)
        elif user_profile['level'] == 'Level-2':
            access_level = random.choice(['Level-1', 'Level-2'])
            transaction_amount = None if random.random() > 0.4 else random.randint(1000, 100000)
        else:
            access_level = random.choice(['Level-1', 'Level-2', 'Level-3'])
            transaction_amount = None if random.random() > 0.5 else random.randint(5000, 500000)
        
        ip_address = f"{location_ips[user_profile['location']]}.{random.randint(10, 250)}"
        duration = random.randint(2, 45)
        device_id = random.choice(devices[:15])
        
        events.append({
            'event_id': f'EVT{event_id:05d}',
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'user_id': user_id,
            'user_name': user_profile['name'],
            'department': user_profile['department'],
            'event_type': event_type,
            'location': user_profile['location'],
            'access_level': access_level,
            'transaction_amount': transaction_amount,
            'ip_address': ip_address,
            'duration_mins': duration,
            'device_id': device_id,
            'status': 'success',
            'is_anomaly': 0,
            'anomaly_type': None
        })
        event_id += 1

    # ========== GENERATE ANOMALIES (52 rows - 15%) ==========

    # Anomaly Type 1: Unusual Time Access (12 events)
    for i in range(12):
        user_id = random.choice(list(normal_users.keys()))
        user_profile = normal_users[user_id]
        
        if user_profile['shift'] == 'Day':
            hour = random.randint(2, 5)
        else:
            hour = random.randint(14, 17)
        
        timestamp = start_date + timedelta(days=random.randint(0, 45), hours=hour, minutes=random.randint(0, 59))
        
        events.append({
            'event_id': f'EVT{event_id:05d}',
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'user_id': user_id,
            'user_name': user_profile['name'],
            'department': user_profile['department'],
            'event_type': 'database_access',
            'location': user_profile['location'],
            'access_level': 'Level-3',
            'transaction_amount': None,
            'ip_address': f"192.168.{random.randint(1,8)}.{random.randint(10, 250)}",
            'duration_mins': random.randint(45, 120),
            'device_id': random.choice(devices),
            'status': 'success',
            'is_anomaly': 1,
            'anomaly_type': 'Unusual Time Access'
        })
        event_id += 1

    # Anomaly Type 2: Geographic Impossibility (10 events)
    for i in range(10):
        user_id = random.choice(list(normal_users.keys()))
        user_profile = normal_users[user_id]
        
        base_time = start_date + timedelta(days=random.randint(0, 45), hours=random.randint(9, 17))
        
        # First login from home location
        events.append({
            'event_id': f'EVT{event_id:05d}',
            'timestamp': base_time.strftime('%Y-%m-%d %H:%M:%S'),
            'user_id': user_id,
            'user_name': user_profile['name'],
            'department': user_profile['department'],
            'event_type': 'login',
            'location': user_profile['location'],
            'access_level': user_profile['level'],
            'transaction_amount': None,
            'ip_address': f"192.168.{random.randint(1,8)}.{random.randint(10, 250)}",
            'duration_mins': 5,
            'device_id': random.choice(devices),
            'status': 'success',
            'is_anomaly': 0,
            'anomaly_type': None
        })
        event_id += 1
        
        # Second login from impossible location 30 mins later
        impossible_time = base_time + timedelta(minutes=30)
        impossible_locations = ['Beijing', 'Karachi', 'Dubai', 'Singapore', 'London']
        
        events.append({
            'event_id': f'EVT{event_id:05d}',
            'timestamp': impossible_time.strftime('%Y-%m-%d %H:%M:%S'),
            'user_id': user_id,
            'user_name': user_profile['name'],
            'department': user_profile['department'],
            'event_type': 'login',
            'location': random.choice(impossible_locations),
            'access_level': user_profile['level'],
            'transaction_amount': None,
            'ip_address': f"{random.randint(50,200)}.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}",
            'duration_mins': 8,
            'device_id': random.choice(devices[15:]),
            'status': 'success',
            'is_anomaly': 1,
            'anomaly_type': 'Geographic Impossibility'
        })
        event_id += 1

    # Anomaly Type 3: Failed Login Attempts (8 events with multiple attempts each)
    for i in range(8):
        user_id = random.choice(list(normal_users.keys()))
        user_profile = normal_users[user_id]
        
        base_time = start_date + timedelta(days=random.randint(0, 45), hours=random.randint(0, 23))
        
        for attempt in range(random.randint(5, 10)):
            timestamp = base_time + timedelta(minutes=attempt * 2)
            events.append({
                'event_id': f'EVT{event_id:05d}',
                'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'user_id': user_id,
                'user_name': user_profile['name'],
                'department': user_profile['department'],
                'event_type': 'login',
                'location': random.choice(['Unknown', 'Proxy Server']),
                'access_level': user_profile['level'],
                'transaction_amount': None,
                'ip_address': f"{random.randint(10,100)}.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}",
                'duration_mins': 0,
                'device_id': 'DEV_UNKNOWN',
                'status': 'failed',
                'is_anomaly': 1,
                'anomaly_type': 'Brute Force Attack'
            })
            event_id += 1

    # Anomaly Type 4: Unusual Transaction Amount (6 events)
    for i in range(6):
        user_id = random.choice(list(normal_users.keys()))
        user_profile = normal_users[user_id]
        
        timestamp = start_date + timedelta(days=random.randint(0, 45), hours=random.randint(9, 18))
        
        events.append({
            'event_id': f'EVT{event_id:05d}',
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'user_id': user_id,
            'user_name': user_profile['name'],
            'department': user_profile['department'],
            'event_type': 'data_export',
            'location': user_profile['location'],
            'access_level': user_profile['level'],
            'transaction_amount': random.randint(5000000, 10000000),
            'ip_address': f"192.168.{random.randint(1,8)}.{random.randint(10, 250)}",
            'duration_mins': random.randint(60, 180),
            'device_id': random.choice(devices),
            'status': 'success',
            'is_anomaly': 1,
            'anomaly_type': 'Unusual Transaction'
        })
        event_id += 1

    # Anomaly Type 5: Privilege Escalation (5 events)
    for i in range(5):
        level1_users = [uid for uid, profile in normal_users.items() if profile['level'] == 'Level-1']
        user_id = random.choice(level1_users)
        user_profile = normal_users[user_id]
        
        timestamp = start_date + timedelta(days=random.randint(0, 45), hours=random.randint(9, 18))
        
        events.append({
            'event_id': f'EVT{event_id:05d}',
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'user_id': user_id,
            'user_name': user_profile['name'],
            'department': user_profile['department'],
            'event_type': 'database_access',
            'location': user_profile['location'],
            'access_level': 'Level-3',
            'transaction_amount': None,
            'ip_address': f"192.168.{random.randint(1,8)}.{random.randint(10, 250)}",
            'duration_mins': random.randint(20, 90),
            'device_id': random.choice(devices),
            'status': 'success',
            'is_anomaly': 1,
            'anomaly_type': 'Privilege Escalation'
        })
        event_id += 1

    # Anomaly Type 6: High-Frequency API Calls (5 events with many calls)
    for i in range(5):
        user_id = random.choice(list(normal_users.keys()))
        user_profile = normal_users[user_id]
        
        base_time = start_date + timedelta(days=random.randint(0, 45), hours=random.randint(9, 18))
        
        for call in range(random.randint(50, 100)):
            timestamp = base_time + timedelta(seconds=call * random.randint(20, 60))
            events.append({
                'event_id': f'EVT{event_id:05d}',
                'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'user_id': user_id,
                'user_name': user_profile['name'],
                'department': user_profile['department'],
                'event_type': 'api_call',
                'location': user_profile['location'],
                'access_level': user_profile['level'],
                'transaction_amount': None,
                'ip_address': f"192.168.{random.randint(1,8)}.{random.randint(10, 250)}",
                'duration_mins': 1,
                'device_id': random.choice(devices),
                'status': 'success',
                'is_anomaly': 1,
                'anomaly_type': 'API Abuse'
            })
            event_id += 1

    # Create DataFrame and sort by timestamp
    df = pd.DataFrame(events)
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    return df


if __name__ == "__main__":
    df = generate_event_logs()
    df.to_csv('event_logs.csv', index=False)
    
    print(f"✅ Generated {len(df)} event logs")
    print(f"\n📊 Normal Events: {len(df[df['is_anomaly'] == 0])} ({len(df[df['is_anomaly'] == 0])/len(df)*100:.1f}%)")
    print(f"📊 Anomalous Events: {len(df[df['is_anomaly'] == 1])} ({len(df[df['is_anomaly'] == 1])/len(df)*100:.1f}%)")
    print("\n🎯 Event Types:")
    print(df['event_type'].value_counts())
    print("\n⚠️ Anomaly Types:")
    print(df[df['is_anomaly'] == 1]['anomaly_type'].value_counts())
