"""
NATGRID Entity Master Data Generator
Generates entity_master.csv with 75 entities across threat levels (RED, ORANGE, PINK)
"""

import pandas as pd
import random
from datetime import datetime, timedelta


def generate_entity_master():
    random.seed(42)
    
    entities = []
    entity_id = 1

    # ========== PINK (LOW) THREAT ENTITIES (38 rows - 50%) ==========
    
    # Politicians
    politicians = [
        'Rahul Mehra', 'Sunita Kapoor', 'Arvind Jha', 'Meena Desai', 'Suresh Patil',
        'Kavita Singh', 'Ravi Sharma', 'Neelam Gupta'
    ]

    for name in politicians:
        entities.append({
            'entity_id': f'ENT{entity_id:04d}',
            'entity_name': name,
            'entity_type': 'PERSON',
            'threat_level': 'PINK',
            'threat_score': random.randint(5, 25),
            'known_affiliations': 'Member of Parliament / State Assembly',
            'last_updated': (datetime.now() - timedelta(days=random.randint(1, 30))).strftime('%Y-%m-%d'),
            'notes': 'Public representative, regular monitoring'
        })
        entity_id += 1

    # Business Leaders
    business_leaders = [
        'Rajesh Mehta', 'Priya Malhotra', 'Amit Shah', 'Deepak Agarwal', 'Sangeeta Rao'
    ]

    for name in business_leaders:
        entities.append({
            'entity_id': f'ENT{entity_id:04d}',
            'entity_name': name,
            'entity_type': 'PERSON',
            'threat_level': 'PINK',
            'threat_score': random.randint(5, 25),
            'known_affiliations': f'CEO of {random.choice(["Tech Solutions", "Global Ventures", "Digital Enterprises"])}',
            'last_updated': (datetime.now() - timedelta(days=random.randint(1, 30))).strftime('%Y-%m-%d'),
            'notes': 'Corporate leader, standard background check cleared'
        })
        entity_id += 1

    # Athletes
    athletes = ['Vikas Kohli', 'Anjali Yadav', 'Rohit Kumar', 'Saina Verma']

    for name in athletes:
        entities.append({
            'entity_id': f'ENT{entity_id:04d}',
            'entity_name': name,
            'entity_type': 'PERSON',
            'threat_level': 'PINK',
            'threat_score': random.randint(5, 20),
            'known_affiliations': f'{random.choice(["Cricket", "Badminton", "Wrestling", "Hockey"])} player',
            'last_updated': (datetime.now() - timedelta(days=random.randint(1, 30))).strftime('%Y-%m-%d'),
            'notes': 'Sports personality, regular international travel'
        })
        entity_id += 1

    # Journalists
    journalists = ['Arun Chatterjee', 'Sneha Nair', 'Karan Bajaj', 'Pooja Reddy']

    for name in journalists:
        entities.append({
            'entity_id': f'ENT{entity_id:04d}',
            'entity_name': name,
            'entity_type': 'PERSON',
            'threat_level': 'PINK',
            'threat_score': random.randint(10, 25),
            'known_affiliations': f'Senior Editor at {random.choice(["Times Network", "NDTV", "India Today", "The Hindu"])}',
            'last_updated': (datetime.now() - timedelta(days=random.randint(1, 30))).strftime('%Y-%m-%d'),
            'notes': 'Media professional, covers political events'
        })
        entity_id += 1

    # Activists
    activists = ['Madhav Iyer', 'Ritu Bhatia', 'Sameer Khan']

    for name in activists:
        entities.append({
            'entity_id': f'ENT{entity_id:04d}',
            'entity_name': name,
            'entity_type': 'PERSON',
            'threat_level': 'PINK',
            'threat_score': random.randint(15, 30),
            'known_affiliations': f'{random.choice(["Environmental", "Human Rights", "Education"])} activist',
            'last_updated': (datetime.now() - timedelta(days=random.randint(1, 30))).strftime('%Y-%m-%d'),
            'notes': 'Social activist, organizes peaceful demonstrations'
        })
        entity_id += 1

    # Regular Organizations
    regular_orgs = [
        'InfoTech Solutions Pvt Ltd', 'Global Consulting Group', 'Digital Innovation Labs',
        'Smart Systems India', 'Tech Dynamics Corporation', 'India Business Council',
        'Mumbai Chamber of Commerce', 'National Sports Federation'
    ]

    for org in regular_orgs:
        entities.append({
            'entity_id': f'ENT{entity_id:04d}',
            'entity_name': org,
            'entity_type': 'ORGANIZATION',
            'threat_level': 'PINK',
            'threat_score': random.randint(5, 20),
            'known_affiliations': 'Registered company, tax compliant',
            'last_updated': (datetime.now() - timedelta(days=random.randint(1, 30))).strftime('%Y-%m-%d'),
            'notes': 'Legitimate business operations, regular audits'
        })
        entity_id += 1

    # ========== ORANGE (MEDIUM) THREAT ENTITIES (22 rows - 30%) ==========

    # Border area residents
    border_residents = [
        'Ashfaq Ali', 'Kuldeep Singh Randhawa', 'Zakir Hussain', 'Gurpreet Kaur',
        'Mohammad Iqbal', 'Harbans Lal'
    ]

    for name in border_residents:
        entities.append({
            'entity_id': f'ENT{entity_id:04d}',
            'entity_name': name,
            'entity_type': 'PERSON',
            'threat_level': 'ORANGE',
            'threat_score': random.randint(35, 55),
            'known_affiliations': f'Resident of {random.choice(["Wagah", "Attari", "Kashmir Valley", "Gujarat border region"])}',
            'last_updated': (datetime.now() - timedelta(days=random.randint(1, 20))).strftime('%Y-%m-%d'),
            'notes': 'Border area resident, frequent cross-border family visits, routine monitoring'
        })
        entity_id += 1

    # Companies under investigation
    investigation_companies = [
        'Coastal Trading Corporation', 'Border Import-Export Ltd', 'Kashmir Valley Traders',
        'Northeast Logistics Services', 'Gujarat Maritime Solutions'
    ]

    for org in investigation_companies:
        entities.append({
            'entity_id': f'ENT{entity_id:04d}',
            'entity_name': org,
            'entity_type': 'ORGANIZATION',
            'threat_level': 'ORANGE',
            'threat_score': random.randint(40, 60),
            'known_affiliations': 'Under customs investigation for irregular documentation',
            'last_updated': (datetime.now() - timedelta(days=random.randint(1, 20))).strftime('%Y-%m-%d'),
            'notes': 'Suspicious transaction patterns detected, enhanced monitoring'
        })
        entity_id += 1

    # Protest organizers
    protest_organizers = [
        'Farmers Unity Forum', 'Youth Rights Collective', 'Workers Welfare Association',
        'Student Democratic Front'
    ]

    for org in protest_organizers:
        entities.append({
            'entity_id': f'ENT{entity_id:04d}',
            'entity_name': org,
            'entity_type': 'ORGANIZATION',
            'threat_level': 'ORANGE',
            'threat_score': random.randint(35, 50),
            'known_affiliations': 'Organizes public demonstrations and protests',
            'last_updated': (datetime.now() - timedelta(days=random.randint(1, 20))).strftime('%Y-%m-%d'),
            'notes': 'Legal organization, some protests turned violent in past'
        })
        entity_id += 1

    # Individuals with concerning contacts
    concerning_contacts = ['Bilal Ahmed', 'Tariq Siddiqui', 'Javed Mir', 'Altaf Sheikh']

    for name in concerning_contacts:
        entities.append({
            'entity_id': f'ENT{entity_id:04d}',
            'entity_name': name,
            'entity_type': 'PERSON',
            'threat_level': 'ORANGE',
            'threat_score': random.randint(45, 65),
            'known_affiliations': 'Known contacts with individuals under surveillance',
            'last_updated': (datetime.now() - timedelta(days=random.randint(1, 20))).strftime('%Y-%m-%d'),
            'notes': 'Indirect links to concerning elements, ongoing assessment'
        })
        entity_id += 1

    # Financial entities
    financial_entities = ['Al-Barkat Money Exchange', 'Kashmir Hawala Network']

    for org in financial_entities:
        entities.append({
            'entity_id': f'ENT{entity_id:04d}',
            'entity_name': org,
            'entity_type': 'ORGANIZATION',
            'threat_level': 'ORANGE',
            'threat_score': random.randint(50, 65),
            'known_affiliations': 'Informal money transfer network',
            'last_updated': (datetime.now() - timedelta(days=random.randint(1, 20))).strftime('%Y-%m-%d'),
            'notes': 'Unregulated financial activities, investigating fund sources'
        })
        entity_id += 1

    # ========== RED (HIGH) THREAT ENTITIES (15 rows - 20%) ==========

    # Known criminals
    criminals = ['Abdul Karim', 'Mohammad Bashir', 'Rashid Ahmed', 'Zaheer Khan']

    for name in criminals:
        entities.append({
            'entity_id': f'ENT{entity_id:04d}',
            'entity_name': name,
            'entity_type': 'PERSON',
            'threat_level': 'RED',
            'threat_score': random.randint(80, 95),
            'known_affiliations': random.choice(['Suspected ISI operative', 'Known terror module member', 'Wanted for border infiltration']),
            'last_updated': (datetime.now() - timedelta(days=random.randint(1, 10))).strftime('%Y-%m-%d'),
            'notes': 'Active surveillance, multiple intelligence reports linking to terrorist activities'
        })
        entity_id += 1

    # Terror operatives
    terror_operatives = ['Imran Sheikh', 'Tariq Mahmood', 'Junaid Afridi']

    for name in terror_operatives:
        entities.append({
            'entity_id': f'ENT{entity_id:04d}',
            'entity_name': name,
            'entity_type': 'PERSON',
            'threat_level': 'RED',
            'threat_score': random.randint(85, 100),
            'known_affiliations': random.choice(['Lashkar-e-Taiba module', 'Jaish handler', 'Hizbul coordinator']),
            'last_updated': (datetime.now() - timedelta(days=random.randint(1, 10))).strftime('%Y-%m-%d'),
            'notes': 'Red corner notice issued, believed to be coordinating attacks from across border'
        })
        entity_id += 1

    # Criminal organizations
    criminal_orgs = [
        'Lashkar Network Kashmir Unit', 'Jaish Module Pulwama Cell', 'Hizbul Mujahideen Operative Cell',
        'ISI Karachi Operations', 'D-Company Dubai Branch'
    ]

    for org in criminal_orgs:
        entities.append({
            'entity_id': f'ENT{entity_id:04d}',
            'entity_name': org,
            'entity_type': 'ORGANIZATION',
            'threat_level': 'RED',
            'threat_score': random.randint(90, 100),
            'known_affiliations': 'Proscribed terrorist organization under UAPA',
            'last_updated': (datetime.now() - timedelta(days=random.randint(1, 10))).strftime('%Y-%m-%d'),
            'notes': 'Designated terror group, active recruitment and funding operations'
        })
        entity_id += 1

    # Arms dealers
    arms_dealers = ['Khalid Arms Syndicate', 'Border Weapon Smuggling Network']

    for org in arms_dealers:
        entities.append({
            'entity_id': f'ENT{entity_id:04d}',
            'entity_name': org,
            'entity_type': 'ORGANIZATION',
            'threat_level': 'RED',
            'threat_score': random.randint(85, 98),
            'known_affiliations': 'Illegal arms trafficking network',
            'last_updated': (datetime.now() - timedelta(days=random.randint(1, 10))).strftime('%Y-%m-%d'),
            'notes': 'Multiple seizures linked to this network, cross-border smuggling operations'
        })
        entity_id += 1

    # Drug cartels
    entities.append({
        'entity_id': f'ENT{entity_id:04d}',
        'entity_name': 'Golden Triangle Connection',
        'entity_type': 'ORGANIZATION',
        'threat_level': 'RED',
        'threat_score': 95,
        'known_affiliations': 'International drug trafficking cartel',
        'last_updated': (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d'),
        'notes': 'Heroin smuggling through Indian ports, links to terror financing'
    })

    # Create DataFrame
    df = pd.DataFrame(entities)
    return df


if __name__ == "__main__":
    df = generate_entity_master()
    df.to_csv('entity_master.csv', index=False)
    
    print(f"✅ Generated {len(df)} entities")
    print("\n📊 Threat Level Distribution:")
    print(df['threat_level'].value_counts())
    print("\n📊 Entity Type Distribution:")
    print(df['entity_type'].value_counts())
    print("\n📊 Average Threat Scores by Level:")
    print(df.groupby('threat_level')['threat_score'].mean().round(1))
