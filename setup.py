#!/usr/bin/env python
"""
NATGRID Intelligence Fusion System - Setup Script
Run this script to generate data and verify installation
"""

import os
import sys

def main():
    print("=" * 60)
    print("NATGRID Intelligence Fusion System - Setup")
    print("=" * 60)
    
    # Get project directory
    project_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(project_dir, 'data')
    
    # Change to project directory
    os.chdir(project_dir)
    sys.path.insert(0, project_dir)
    
    # Check and generate data
    print("\n📁 Checking data files...")
    
    from data.generate_reports import generate_intelligence_reports
    from data.generate_events import generate_event_logs
    from data.generate_entities import generate_entity_master
    
    # Generate reports
    reports_path = os.path.join(data_dir, 'intelligence_reports.csv')
    if not os.path.exists(reports_path):
        print("   Generating intelligence_reports.csv...")
        df = generate_intelligence_reports()
        df.to_csv(reports_path, index=False)
        print(f"   ✅ Generated {len(df)} reports")
    else:
        print("   ✅ intelligence_reports.csv exists")
    
    # Generate events
    events_path = os.path.join(data_dir, 'event_logs.csv')
    if not os.path.exists(events_path):
        print("   Generating event_logs.csv...")
        df = generate_event_logs()
        df.to_csv(events_path, index=False)
        print(f"   ✅ Generated {len(df)} events")
    else:
        print("   ✅ event_logs.csv exists")
    
    # Generate entities
    entities_path = os.path.join(data_dir, 'entity_master.csv')
    if not os.path.exists(entities_path):
        print("   Generating entity_master.csv...")
        df = generate_entity_master()
        df.to_csv(entities_path, index=False)
        print(f"   ✅ Generated {len(df)} entities")
    else:
        print("   ✅ entity_master.csv exists")
    
    # Verify dependencies
    print("\n📦 Checking dependencies...")
    
    required = ['streamlit', 'pandas', 'numpy', 'plotly', 'sklearn', 'transformers', 'sentence_transformers']
    missing = []
    
    for package in required:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("   Run: pip install -r requirements.txt")
    else:
        print("\n✅ All dependencies installed!")
    
    print("\n" + "=" * 60)
    print("🚀 To start the application, run:")
    print("   streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
