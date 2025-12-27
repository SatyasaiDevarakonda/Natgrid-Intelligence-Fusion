"""
NATGRID Intelligence Reports Data Generator
Generates intelligence_reports.csv with 180 reports across priority levels
Enhanced with better entity extraction support and more realistic data
"""

import pandas as pd
import random
from datetime import datetime, timedelta


def generate_intelligence_reports():
    random.seed(42)

    reports = []

    sources = ['OSINT', 'Media Monitoring', 'Field Report', 'Cyber Intel', 'Social Media', 'Tip-off', 'Satellite', 'Informant']
    locations = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad', 'Pune', 'Ahmedabad', 'Jaipur', 'Lucknow', 
                 'Wagah Border', 'Attari', 'Kashmir', 'Gujarat Coast', 'Mundra Port', 'Kandla Port', 'Northeast Region']

    report_id = 1

    # ========== REGULAR NEWS (108 rows - 60%) ==========
    regular_news_templates = [
        ("Sports", "LOW", "Cricket match between India and {country} scheduled at {location}. Expected crowd of 50000 spectators. Security arrangements finalized by local police. Sports coordinator Vikas Kohli overseeing preparations."),
        ("Politics", "LOW", "Chief Minister Rajesh Mehta inaugurated new metro line in {location}. Project worth 5000 crore rupees completed ahead of schedule. Infrastructure Minister Sunita Kapoor was also present at the ceremony."),
        ("Business", "LOW", "Tech company InfoTech Solutions Pvt Ltd announced expansion plans in {location}. Will create 2000 new jobs in next fiscal year. CEO Priya Malhotra addressed shareholders at annual meeting."),
        ("Technology", "LOW", "IIT {location} researchers develop new AI algorithm for agricultural prediction. Paper published in international journal. Research funded by Digital Innovation Labs in partnership with Tech Dynamics Corporation."),
        ("Entertainment", "LOW", "Bollywood production house announced shooting schedule at {location}. Film directed by renowned filmmaker expected to release next year. Mumbai Chamber of Commerce supporting local film initiatives."),
        ("Weather", "LOW", "Meteorological department predicts heavy rainfall in {location} region. Schools advised to remain closed for two days. Local authorities coordinating with National Sports Federation for event rescheduling."),
        ("Education", "LOW", "CBSE announces board exam dates for 2025. Over 2 million students expected to appear nationwide. Education Minister Meena Desai released official notification from {location}."),
        ("Health", "LOW", "New hospital facility inaugurated in {location} with 500 bed capacity. Equipped with modern medical equipment. India Business Council provided partial funding for the healthcare initiative."),
        ("Infrastructure", "LOW", "Highway construction project between {location} and Delhi nearing completion. Expected to reduce travel time by 3 hours. Global Consulting Group provided project management services."),
        ("Social", "LOW", "Annual cultural festival begins in {location}. Artists from 15 states participating in week-long event. Smart Systems India providing technical infrastructure for the venue."),
    ]

    countries_friendly = ['Australia', 'England', 'Sri Lanka', 'Bangladesh', 'New Zealand', 'South Africa']

    for i in range(108):
        template = random.choice(regular_news_templates)
        date = datetime(2024, 10, 1) + timedelta(days=random.randint(0, 75))
        
        report_text = template[2].format(
            country=random.choice(countries_friendly),
            location=random.choice(locations[:10])
        )
        
        reports.append({
            'report_id': f'RPT{report_id:04d}',
            'date': date.strftime('%Y-%m-%d'),
            'source': random.choice(sources[:5]),
            'category': template[0],
            'priority': template[1],
            'report_text': report_text,
            'classification': 'UNCLASSIFIED',
            'confidence_score': random.randint(75, 95)
        })
        report_id += 1

    # ========== BORDERLINE/CONTEXTUAL (45 rows - 25%) ==========
    borderline_templates = [
        ("Civil Unrest", "MEDIUM", "Farmers Unity Forum protest continues in {location} for third consecutive day. Around 500 protesters blocking highway demanding better crop prices. Workers Welfare Association also joined the demonstration. Police monitoring situation closely."),
        ("Border Activity", "MEDIUM", "Increased vehicle movement observed near {location} border checkpoint. Coastal Trading Corporation vehicles flagged for additional inspection. Customs officials report longer than usual queues. Border Import-Export Ltd shipment delayed."),
        ("Accidents", "MEDIUM", "Gas leak reported at industrial facility in {location}. 15 workers hospitalized, area evacuated as precautionary measure. Gujarat Maritime Solutions facility under investigation. Northeast Logistics Services transport suspended temporarily."),
        ("Diplomatic", "MEDIUM", "Delegation from {country} arrives in {location} for trade discussions. Meetings scheduled with commerce ministry officials. Kashmir Valley Traders seeking bilateral agreement amendments."),
        ("Cyber Incident", "MEDIUM", "Government website faced temporary outage in {location} due to technical glitch. Services restored within 2 hours. Youth Rights Collective claiming denial of service but investigation ongoing."),
        ("Public Gathering", "MEDIUM", "Religious congregation expected to draw 10000 people in {location}. Police deploying additional personnel for crowd management. Student Democratic Front organizing peaceful participation."),
        ("Labor Dispute", "MEDIUM", "Workers at manufacturing facility in {location} staging strike over wage disputes. Management in negotiations with union representatives. Al-Barkat Money Exchange handling worker remittances under scrutiny."),
        ("Environmental", "MEDIUM", "Illegal construction activity reported near protected forest area in {location}. Forest department investigating the matter. Kashmir Hawala Network suspected of financing unauthorized construction."),
    ]

    countries_neutral = ['China', 'Pakistan', 'Nepal', 'Myanmar', 'Afghanistan', 'Bangladesh']

    for i in range(45):
        template = random.choice(borderline_templates)
        date = datetime(2024, 10, 1) + timedelta(days=random.randint(0, 75))
        
        report_text = template[2].format(
            location=random.choice(locations),
            country=random.choice(countries_neutral)
        )
        
        reports.append({
            'report_id': f'RPT{report_id:04d}',
            'date': date.strftime('%Y-%m-%d'),
            'source': random.choice(sources),
            'category': template[0],
            'priority': template[1],
            'report_text': report_text,
            'classification': 'RESTRICTED',
            'confidence_score': random.randint(60, 85)
        })
        report_id += 1

    # ========== HIGH PRIORITY THREATS (27 rows - 15%) ==========
    threat_templates = [
        ("Terrorism", "HIGH", "Intelligence indicates Abdul Karim and Mohammad Bashir planning coordinated attack in {location}. Intercepted communications mention explosives procurement. Lashkar Network Kashmir Unit involvement suspected. ISI Karachi Operations may be providing logistical support."),
        ("Smuggling", "HIGH", "Customs officials seized 50 kg contraband at {location}. Suspect Rashid Ahmed arrested, investigation reveals links to D-Company Dubai Branch cartel. Border Weapon Smuggling Network suspected involvement."),
        ("Cyber Attack", "HIGH", "Sophisticated malware detected targeting critical infrastructure in {location}. Attack vector traced to IP addresses in {country}. Jaish Module Pulwama Cell digital operations wing suspected. Data breach confirmed affecting multiple government systems."),
        ("Espionage", "HIGH", "Foreign national Zaheer Khan under surveillance for suspicious activities near military installation at {location}. Possible intelligence gathering for ISI Karachi Operations. Hizbul Mujahideen Operative Cell contacts identified."),
        ("Terror Financing", "HIGH", "Hawala transactions worth 2 crore traced to Al-Barkat Money Exchange. Financial trail leads to accounts linked with Lashkar Network Kashmir Unit. Tariq Mahmood identified as key facilitator. Kashmir Hawala Network under active monitoring."),
        ("Border Infiltration", "HIGH", "Movement of armed individuals detected near {location}. Thermal imaging shows group of 8-10 persons attempting border crossing at night. Imran Sheikh believed to be coordinating operation. Jaish Module Pulwama Cell operatives identified."),
        ("Weapon Smuggling", "HIGH", "Truck intercepted at {location} checkpoint carrying illegal firearms. Driver admits weapons meant for Hizbul Mujahideen Operative Cell. Khalid Arms Syndicate involvement confirmed. Golden Triangle Connection providing funding."),
        ("Radicalization", "HIGH", "Online propaganda spread by Jaish Module Pulwama Cell targeting youth in {location} region. Social media accounts promoting extremist ideology identified. Junaid Afridi suspected of content creation. D-Company Dubai Branch funding operation."),
        ("Maritime Threat", "HIGH", "Unidentified vessel spotted 15 nautical miles from {location} coast. Not responding to coast guard signals, possible smuggling attempt. Border Weapon Smuggling Network suspected. Golden Triangle Connection drug shipment likely."),
    ]

    for i in range(27):
        template = random.choice(threat_templates)
        date = datetime(2024, 10, 1) + timedelta(days=random.randint(0, 75))
        
        report_text = template[2].format(
            location=random.choice(locations),
            country=random.choice(['Pakistan', 'China', 'Unknown'])
        )
        
        reports.append({
            'report_id': f'RPT{report_id:04d}',
            'date': date.strftime('%Y-%m-%d'),
            'source': random.choice(sources),
            'category': template[0],
            'priority': template[1],
            'report_text': report_text,
            'classification': 'SECRET',
            'confidence_score': random.randint(70, 95)
        })
        report_id += 1

    # Create DataFrame and shuffle
    df = pd.DataFrame(reports)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df


if __name__ == "__main__":
    df = generate_intelligence_reports()
    df.to_csv('intelligence_reports.csv', index=False)
    
    print(f"✅ Generated {len(df)} intelligence reports")
    print("\n📊 Category Distribution:")
    print(df['category'].value_counts())
    print("\n📊 Priority Distribution:")
    print(df['priority'].value_counts())
    print("\n📊 Classification Distribution:")
    print(df['classification'].value_counts())
    print("\n📊 Average Confidence by Priority:")
    print(df.groupby('priority')['confidence_score'].mean().round(1))
