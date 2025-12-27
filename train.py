#!/usr/bin/env python
"""
NATGRID Model Training Script
Trains and saves all models required for the Intelligence Fusion System

Models trained:
1. Anomaly Detection (Isolation Forest)
2. Semantic Search Index (Sentence Embeddings)
3. NER Model validation
4. LLM initialization and caching

Usage:
    python train.py --all           # Train all models
    python train.py --anomaly       # Train only anomaly detection
    python train.py --embeddings    # Train only search embeddings
    python train.py --validate-ner  # Validate NER model
    python train.py --init-llm      # Initialize and cache LLM
"""

import os
import sys
import argparse
import json
import pickle
import time
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

# Add project directory to path
PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR))

from config import get_config, check_gpu_availability


def print_header(title: str):
    """Print a formatted header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def print_metrics(metrics: dict, indent: int = 2):
    """Print metrics in a formatted way"""
    prefix = " " * indent
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{prefix}{key}: {value:.4f}")
        elif isinstance(value, dict):
            print(f"{prefix}{key}:")
            print_metrics(value, indent + 2)
        elif isinstance(value, np.ndarray):
            print(f"{prefix}{key}: {value.tolist()}")
        else:
            print(f"{prefix}{key}: {value}")


def train_anomaly_detection(config, save_model: bool = True) -> dict:
    """
    Train the Isolation Forest anomaly detection model
    
    Args:
        config: Configuration object
        save_model: Whether to save the trained model
        
    Returns:
        Training metrics dictionary
    """
    print_header("TRAINING ANOMALY DETECTION MODEL")
    
    from utils.anomaly_module import AnomalyDetector
    
    # Load event logs
    events_path = config.data_dir / "event_logs.csv"
    if not events_path.exists():
        raise FileNotFoundError(f"Event logs not found at {events_path}")
    
    print(f"Loading data from {events_path}...")
    events_df = pd.read_csv(events_path)
    print(f"  Total events: {len(events_df)}")
    print(f"  Normal events: {len(events_df[events_df['is_anomaly'] == 0])}")
    print(f"  Anomalous events: {len(events_df[events_df['is_anomaly'] == 1])}")
    
    # Initialize detector with n_estimators
    print(f"\nInitializing Isolation Forest...")
    print(f"  Contamination: {config.anomaly_contamination}")
    print(f"  N Estimators: {config.anomaly_n_estimators}")
    print(f"  Random Seed: {config.random_seed}")
    
    # FIXED: Pass n_estimators during initialization
    detector = AnomalyDetector(
        contamination=config.anomaly_contamination,
        n_estimators=config.anomaly_n_estimators,
        random_state=config.random_seed
    )
    
    # Train the model
    print("\nTraining model...")
    start_time = time.time()
    detector.fit(events_df)
    train_time = time.time() - start_time
    print(f"  Training completed in {train_time:.2f} seconds")
    
    # Evaluate
    print("\nEvaluating model...")
    metrics = detector.evaluate(events_df)
    
    print("\n📊 EVALUATION METRICS:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}" if metrics['precision'] else "  Precision: N/A")
    print(f"  Recall: {metrics['recall']:.4f}" if metrics['recall'] else "  Recall: N/A")
    print(f"  F1 Score: {metrics['f1_score']:.4f}" if metrics['f1_score'] else "  F1 Score: N/A")
    print(f"  ROC AUC: {metrics['roc_auc']:.4f}" if metrics['roc_auc'] else "  ROC AUC: N/A")
    
    print("\n📊 CONFUSION MATRIX:")
    cm = metrics['confusion_matrix']
    print(f"  True Negatives: {cm[0][0]}")
    print(f"  False Positives: {cm[0][1]}")
    print(f"  False Negatives: {cm[1][0]}")
    print(f"  True Positives: {cm[1][1]}")
    
    # Save model
    if save_model:
        model_path = config.models_dir / "anomaly_detector.pkl"
        print(f"\nSaving model to {model_path}...")
        detector.save_model(str(model_path))
        print("  ✅ Model saved successfully")
    
    # Get top anomalies
    print("\n🚨 TOP 5 DETECTED ANOMALIES:")
    top_anomalies = detector.get_top_anomalies(events_df, top_k=5)
    for i, (_, row) in enumerate(top_anomalies.iterrows(), 1):
        print(f"  {i}. {row['event_id']} | Risk: {row['risk_score']:.1f} | User: {row['user_id']} | Type: {row['event_type']}")
    
    return {
        'train_time': train_time,
        'accuracy': metrics['accuracy'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1_score': metrics['f1_score'],
        'roc_auc': metrics['roc_auc'],
        'confusion_matrix': cm.tolist()
    }


def train_search_embeddings(config, save_index: bool = True) -> dict:
    """
    Generate and save search embeddings for intelligence reports
    
    Args:
        config: Configuration object
        save_index: Whether to save the index
        
    Returns:
        Training metrics dictionary
    """
    print_header("TRAINING SEMANTIC SEARCH INDEX")
    
    from utils.embedding_module import SemanticSearchEngine
    
    # Load reports
    reports_path = config.data_dir / "intelligence_reports.csv"
    if not reports_path.exists():
        raise FileNotFoundError(f"Reports not found at {reports_path}")
    
    print(f"Loading data from {reports_path}...")
    reports_df = pd.read_csv(reports_path)
    print(f"  Total reports: {len(reports_df)}")
    
    # Initialize search engine
    print(f"\nInitializing embedding model: {config.embedding_model}")
    start_time = time.time()
    engine = SemanticSearchEngine(model_name=config.embedding_model)
    init_time = time.time() - start_time
    print(f"  Model loaded in {init_time:.2f} seconds")
    
    # Index documents
    print("\nIndexing documents...")
    start_time = time.time()
    engine.index_documents(
        documents=reports_df['report_text'].tolist(),
        document_ids=reports_df['report_id'].tolist()
    )
    index_time = time.time() - start_time
    print(f"  Indexed {len(reports_df)} documents in {index_time:.2f} seconds")
    print(f"  Embedding dimension: {engine.embeddings.shape[1]}")
    
    # Test search
    print("\n📊 SEARCH QUALITY TESTS:")
    test_queries = [
        ("weapons smuggling border", ["Smuggling", "Weapon Smuggling", "Border Infiltration"]),
        ("cyber attack malware", ["Cyber Attack", "Cyber Incident"]),
        ("terror financing hawala", ["Terror Financing"]),
        ("cricket match sports", ["Sports"]),
        ("protest farmers", ["Civil Unrest", "Labor Dispute"])
    ]
    
    total_correct = 0
    total_tests = 0
    
    for query, expected_categories in test_queries:
        results = engine.search(query, top_k=5)
        
        # Check if any result matches expected categories
        result_ids = [r['document_id'] for r in results]
        result_categories = reports_df[reports_df['report_id'].isin(result_ids)]['category'].tolist()
        
        correct = any(cat in expected_categories for cat in result_categories)
        total_correct += 1 if correct else 0
        total_tests += 1
        
        status = "✅" if correct else "❌"
        print(f"  {status} Query: '{query}'")
        print(f"       Top result: {results[0]['document_id']} (similarity: {results[0]['similarity']:.3f})")
    
    search_accuracy = total_correct / total_tests
    print(f"\n  Search Accuracy: {search_accuracy:.1%} ({total_correct}/{total_tests} queries)")
    
    # Test clustering
    print("\n📊 DOCUMENT CLUSTERING:")
    clusters = engine.cluster_documents(n_clusters=5)
    for cluster_id, doc_ids in clusters.items():
        # Get dominant category
        cluster_categories = reports_df[reports_df['report_id'].isin(doc_ids)]['category'].mode()
        dominant_cat = cluster_categories.iloc[0] if len(cluster_categories) > 0 else "Unknown"
        print(f"  Cluster {cluster_id}: {len(doc_ids)} docs (dominant: {dominant_cat})")
    
    # Save index
    if save_index:
        index_path = config.models_dir / "search_index.pkl"
        print(f"\nSaving index to {index_path}...")
        engine.save_index(str(index_path))
        print("  ✅ Index saved successfully")
    
    return {
        'init_time': init_time,
        'index_time': index_time,
        'num_documents': len(reports_df),
        'embedding_dim': engine.embeddings.shape[1],
        'search_accuracy': search_accuracy
    }


def validate_ner_model(config) -> dict:
    """
    Validate the NER model on sample data
    
    Args:
        config: Configuration object
        
    Returns:
        Validation metrics dictionary
    """
    print_header("VALIDATING NER MODEL")
    
    from utils.ner_module import NERExtractor
    
    # Initialize NER
    print(f"Loading NER model: {config.ner_model}")
    start_time = time.time()
    ner = NERExtractor(model_name=config.ner_model)
    load_time = time.time() - start_time
    print(f"  Model loaded in {load_time:.2f} seconds")
    
    # Test cases with expected entities
    test_cases = [
        {
            "text": "Abdul Karim was spotted in Mumbai near the Lashkar hideout.",
            "expected": {
                "persons": ["Abdul Karim"],
                "organizations": ["Lashkar"],
                "locations": ["Mumbai"]
            }
        },
        {
            "text": "The ISI operative met with D-Company members in Karachi.",
            "expected": {
                "persons": [],
                "organizations": ["ISI", "D-Company"],
                "locations": ["Karachi"]
            }
        },
        {
            "text": "Rajesh Mehta, CEO of InfoTech Solutions, announced expansion in Bangalore.",
            "expected": {
                "persons": ["Rajesh Mehta"],
                "organizations": ["InfoTech Solutions"],
                "locations": ["Bangalore"]
            }
        },
        {
            "text": "Customs officials at Mundra Port seized contraband from Mohammad Bashir.",
            "expected": {
                "persons": ["Mohammad Bashir"],
                "organizations": [],
                "locations": ["Mundra Port"]
            }
        },
        {
            "text": "Chief Minister Priya Sharma inaugurated the new metro line in Delhi.",
            "expected": {
                "persons": ["Priya Sharma"],
                "organizations": [],
                "locations": ["Delhi"]
            }
        }
    ]
    
    print("\n📊 NER VALIDATION RESULTS:")
    
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n  Test {i}: {test['text'][:60]}...")
        
        # Extract entities
        extracted = ner.extract_entities(test["text"])
        expected = test["expected"]
        
        # Calculate metrics for each entity type
        for entity_type in ["persons", "organizations", "locations"]:
            extracted_set = set(e.lower() for e in extracted[entity_type])
            expected_set = set(e.lower() for e in expected[entity_type])
            
            if len(expected_set) > 0 or len(extracted_set) > 0:
                tp = len(extracted_set & expected_set)
                fp = len(extracted_set - expected_set)
                fn = len(expected_set - extracted_set)
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                total_precision += precision
                total_recall += recall
                total_f1 += f1
                
                status = "✅" if f1 >= 0.5 else "⚠️" if f1 > 0 else "❌"
                print(f"       {status} {entity_type}: extracted={list(extracted[entity_type])}, expected={expected[entity_type]}")
    
    # Calculate averages
    num_tests = len(test_cases) * 3  # 3 entity types per test
    avg_precision = total_precision / num_tests
    avg_recall = total_recall / num_tests
    avg_f1 = total_f1 / num_tests
    
    print(f"\n  OVERALL METRICS:")
    print(f"    Precision: {avg_precision:.4f}")
    print(f"    Recall: {avg_recall:.4f}")
    print(f"    F1 Score: {avg_f1:.4f}")
    
    # Test on actual reports
    print("\n📊 TESTING ON ACTUAL REPORTS:")
    reports_path = config.data_dir / "intelligence_reports.csv"
    if reports_path.exists():
        reports_df = pd.read_csv(reports_path)
        high_priority = reports_df[reports_df['priority'] == 'HIGH'].head(3)
        
        for _, row in high_priority.iterrows():
            entities = ner.extract_entities(row['report_text'])
            print(f"\n  Report {row['report_id']} ({row['category']}):")
            print(f"    Persons: {entities['persons'][:5]}")
            print(f"    Organizations: {entities['organizations'][:5]}")
            print(f"    Locations: {entities['locations'][:5]}")
    
    return {
        'load_time': load_time,
        'precision': avg_precision,
        'recall': avg_recall,
        'f1_score': avg_f1,
        'num_test_cases': len(test_cases)
    }


def initialize_llm(config) -> dict:
    """
    Initialize and test the LLM
    
    Args:
        config: Configuration object
        
    Returns:
        Initialization metrics dictionary
    """
    print_header("INITIALIZING LLM")
    
    print(f"LLM Provider: {config.llm_provider}")
    
    # Show GPU info if using local
    if config.llm_provider == "local_gpu":
        print("\nChecking GPU availability...")
        gpu_info = check_gpu_availability()
        
        if gpu_info['cuda_available']:
            print(f"  ✅ GPU Available: {gpu_info['devices'][0]['name']}")
            print(f"  VRAM: {gpu_info['devices'][0]['memory_total']:.1f} GB")
        else:
            print("  ⚠️ GPU not available")
    
    # Initialize LLM
    print(f"\nInitializing {config.llm_provider}...")
    
    start_time = time.time()
    
    from utils.llm_module import get_intelligence_analyzer
    
    # FIXED: Pass only the provider string, not the entire config object
    analyzer = get_intelligence_analyzer(config.llm_provider)
    
    load_time = time.time() - start_time
    print(f"\n  Initialized in {load_time:.2f} seconds")
    
    # Test generation
    print("\n📊 TESTING TEXT GENERATION:")
    
    test_prompt = "Summarize this intelligence report in one paragraph: A suspected terrorist was spotted near the border checkpoint. Customs officials are on high alert."
    
    print(f"  Prompt: {test_prompt[:80]}...")
    
    gen_start = time.time()
    response = analyzer.llm.generate(test_prompt, max_tokens=150, temperature=0.3)
    gen_time = time.time() - gen_start
    
    print(f"  Response ({gen_time:.2f}s):")
    print(f"    {response[:200]}...")
    
    # Calculate tokens per second
    approx_tokens = len(response.split())
    tokens_per_sec = approx_tokens / gen_time if gen_time > 0 else 0
    
    print(f"\n  Generation Speed: ~{tokens_per_sec:.1f} tokens/second")
    print(f"\n  ✅ LLM test successful!")
    
    return {
        'provider': config.llm_provider,
        'load_time': load_time,
        'generation_time': gen_time,
        'tokens_per_second': tokens_per_sec
    }


def save_training_report(metrics: dict, config):
    """Save training metrics to a JSON report"""
    report = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'llm_provider': config.llm_provider,
            'ner_model': config.ner_model,
            'embedding_model': config.embedding_model,
            'anomaly_contamination': config.anomaly_contamination
        },
        'metrics': metrics
    }
    
    report_path = config.models_dir / "training_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Training report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="NATGRID Model Training Script")
    parser.add_argument('--all', action='store_true', help='Train all models')
    parser.add_argument('--anomaly', action='store_true', help='Train anomaly detection model')
    parser.add_argument('--embeddings', action='store_true', help='Train search embeddings')
    parser.add_argument('--validate-ner', action='store_true', help='Validate NER model')
    parser.add_argument('--init-llm', action='store_true', help='Initialize and test LLM')
    parser.add_argument('--no-save', action='store_true', help='Do not save models')
    
    args = parser.parse_args()
    
    # If no specific model selected, train all
    if not any([args.anomaly, args.embeddings, args.validate_ner, args.init_llm]):
        args.all = True
    
    print_header("NATGRID MODEL TRAINING")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Load configuration
        print("\nLoading configuration...")
        config = get_config()
        print("  ✅ Configuration loaded")
        
        all_metrics = {}
        save_models = not args.no_save
        
        # Train anomaly detection
        if args.all or args.anomaly:
            try:
                metrics = train_anomaly_detection(config, save_model=save_models)
                all_metrics['anomaly_detection'] = metrics
            except Exception as e:
                print(f"  ❌ Anomaly detection training failed: {e}")
                all_metrics['anomaly_detection'] = {'error': str(e)}
        
        # Train search embeddings
        if args.all or args.embeddings:
            try:
                metrics = train_search_embeddings(config, save_index=save_models)
                all_metrics['search_embeddings'] = metrics
            except Exception as e:
                print(f"  ❌ Embedding training failed: {e}")
                all_metrics['search_embeddings'] = {'error': str(e)}
        
        # Validate NER
        if args.all or args.validate_ner:
            try:
                metrics = validate_ner_model(config)
                all_metrics['ner_validation'] = metrics
            except Exception as e:
                print(f"  ❌ NER validation failed: {e}")
                all_metrics['ner_validation'] = {'error': str(e)}
        
        # Initialize LLM
        if args.all or args.init_llm:
            try:
                metrics = initialize_llm(config)
                all_metrics['llm_initialization'] = metrics
            except Exception as e:
                print(f"  ❌ LLM initialization failed: {e}")
                all_metrics['llm_initialization'] = {'error': str(e)}
        
        # Save training report
        if save_models:
            save_training_report(all_metrics, config)
        
        # Summary
        print_header("TRAINING COMPLETE")
        print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print("\n📊 SUMMARY:")
        for model_name, metrics in all_metrics.items():
            if 'error' in metrics:
                print(f"  ❌ {model_name}: FAILED - {metrics['error']}")
            else:
                print(f"  ✅ {model_name}: SUCCESS")
        
        print(f"\n📁 Models saved to: {config.models_dir}")
        
    except ValueError as e:
        print(f"\n❌ Configuration Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Training Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()