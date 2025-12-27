🛡️ NATGRID Intelligence Fusion System

AI-powered intelligence analysis platform for security data processing and threat detection.

🚀 Quick Start
# Setup environment
cp .env.example .env


Add one API key in .env:

MISTRAL_API_KEY=your_api_key
LLM_PROVIDER=mistral_api


Install & run:

pip install -r requirements.txt

cd data
python generate_reports.py
python generate_events.py
python generate_entities.py
cd ..

python train.py --all
python -m streamlit run app.py

✨ Core Features

📊 Interactive dashboard & analytics

🔍 Semantic intelligence report search

👤 Entity threat profiling

⚠️ ML-based anomaly detection

🔗 Multi-source intelligence fusion

📁 Project Structure
natgrid_project/
├── app.py
├── train.py
├── config.py
├── requirements.txt
├── data/
│   ├── intelligence_reports.csv
│   ├── event_logs.csv
│   └── entity_master.csv
├── models/
└── utils/

📊 Datasets Used

intelligence_reports.csv – OSINT & field intelligence reports

event_logs.csv – User activity logs with anomalies

entity_master.csv – Persons & organizations with threat levels

🧠 Anomaly Detection

Isolation Forest–based risk scoring:

CRITICAL: >90

HIGH: 70–90

MEDIUM: 50–70

LOW: <50

🔧 Troubleshooting

API error → Check .env and LLM_PROVIDER

Missing modules → pip install -r requirements.txt

Missing data → Run data generation scripts