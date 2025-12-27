"""
NATGRID Anomaly Detection Module
Uses Isolation Forest and feature engineering for detecting suspicious activities
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from typing import Dict, List, Tuple, Optional
import pickle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Anomaly detection for event logs using Isolation Forest"""
    
    def __init__(self, contamination: float = 0.15, n_estimators: int = 100, random_state: int = 42):
        """
        Initialize the anomaly detector
        
        Args:
            contamination: Expected proportion of anomalies in the dataset
            n_estimators: Number of trees in the Isolation Forest
            random_state: Random seed for reproducibility
        """
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.is_fitted = False
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features from raw event logs
        
        Args:
            df: Raw event logs DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        df = df.copy()
        
        # Parse timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Temporal features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
        
        # Encode categorical variables
        categorical_cols = ['location', 'access_level', 'event_type', 'device_id']
        
        for col in categorical_cols:
            if col in df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col].fillna('Unknown'))
                else:
                    # Handle unseen labels
                    df[f'{col}_encoded'] = df[col].apply(
                        lambda x: self.label_encoders[col].transform([x])[0] 
                        if x in self.label_encoders[col].classes_ 
                        else -1
                    )
        
        # IP risk score
        df['ip_risk_score'] = df['ip_address'].apply(
            lambda x: 1 if str(x).startswith('192.168') else 5
        )
        
        # Transaction amount (fill NaN with 0)
        df['transaction_amount_filled'] = df['transaction_amount'].fillna(0)
        df['has_transaction'] = (df['transaction_amount'].notna()).astype(int)
        
        # Duration features
        df['duration_mins'] = df['duration_mins'].fillna(0)
        df['long_duration'] = (df['duration_mins'] > 60).astype(int)
        
        # Status encoding
        df['status_success'] = (df['status'] == 'success').astype(int)
        
        return df
    
    def get_feature_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Extract feature matrix for model training/prediction"""
        self.feature_columns = [
            'hour', 'day_of_week', 'is_weekend', 'is_night',
            'location_encoded', 'access_level_encoded', 'event_type_encoded',
            'ip_risk_score', 'transaction_amount_filled', 'has_transaction',
            'duration_mins', 'long_duration', 'status_success'
        ]
        
        # Only use columns that exist
        available_cols = [col for col in self.feature_columns if col in df.columns]
        
        X = df[available_cols].values
        return X
    
    def fit(self, df: pd.DataFrame) -> 'AnomalyDetector':
        """
        Fit the anomaly detection model
        
        Args:
            df: DataFrame with event logs
            
        Returns:
            Self
        """
        logger.info("Preparing features...")
        df_prepared = self.prepare_features(df)
        
        logger.info("Extracting feature matrix...")
        X = self.get_feature_matrix(df_prepared)
        
        logger.info("Scaling features...")
        X_scaled = self.scaler.fit_transform(X)
        
        logger.info(f"Training Isolation Forest (contamination={self.contamination}, n_estimators={self.n_estimators})...")
        # FIXED: Use self.n_estimators instead of hardcoded value
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=self.n_estimators,
            max_samples='auto',
            n_jobs=-1
        )
        self.model.fit(X_scaled)
        
        self.is_fitted = True
        logger.info("Model training complete")
        
        return self
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict anomalies in event logs
        
        Args:
            df: DataFrame with event logs
            
        Returns:
            DataFrame with anomaly predictions and scores
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        df_prepared = self.prepare_features(df)
        X = self.get_feature_matrix(df_prepared)
        X_scaled = self.scaler.transform(X)
        
        # Predict (-1 = anomaly, 1 = normal)
        predictions = self.model.predict(X_scaled)
        
        # Get anomaly scores (more negative = more anomalous)
        scores = self.model.score_samples(X_scaled)
        
        # Add to dataframe
        df_result = df.copy()
        df_result['predicted_anomaly'] = (predictions == -1).astype(int)
        df_result['anomaly_score'] = scores
        df_result['risk_score'] = ((1 - (scores - scores.min()) / (scores.max() - scores.min())) * 100)
        
        return df_result
    
    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit the model and predict anomalies"""
        self.fit(df)
        return self.predict(df)
    
    def evaluate(self, df: pd.DataFrame, true_labels_col: str = 'is_anomaly') -> Dict:
        """
        Evaluate the model against ground truth labels
        
        Args:
            df: DataFrame with true labels
            true_labels_col: Column name for true labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        df_pred = self.predict(df)
        
        y_true = df[true_labels_col].values
        y_pred = df_pred['predicted_anomaly'].values
        scores = df_pred['anomaly_score'].values
        
        # Classification report
        report = classification_report(y_true, y_pred, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # ROC AUC (using negative scores since more negative = more anomalous)
        try:
            auc = roc_auc_score(y_true, -scores)
            fpr, tpr, thresholds = roc_curve(y_true, -scores)
        except Exception as e:
            logger.warning(f"Could not compute ROC AUC: {e}")
            auc = None
            fpr, tpr, thresholds = None, None, None
        
        return {
            'classification_report': report,
            'confusion_matrix': cm,
            'roc_auc': auc,
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'accuracy': report['accuracy'],
            'precision': report['1']['precision'] if '1' in report else None,
            'recall': report['1']['recall'] if '1' in report else None,
            'f1_score': report['1']['f1-score'] if '1' in report else None
        }
    
    def get_top_anomalies(self, df: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
        """Get the top-k most anomalous events"""
        df_pred = self.predict(df)
        return df_pred.nsmallest(top_k, 'anomaly_score')
    
    def save_model(self, filepath: str):
        """Save the model to disk"""
        data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'contamination': self.contamination,
            'n_estimators': self.n_estimators
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load the model from disk"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.model = data['model']
        self.scaler = data['scaler']
        self.label_encoders = data['label_encoders']
        self.feature_columns = data['feature_columns']
        self.contamination = data['contamination']
        self.n_estimators = data.get('n_estimators', 100)  # Backward compatibility
        self.is_fitted = True
        logger.info(f"Model loaded from {filepath}")


def get_anomaly_detector(contamination: float = 0.15, n_estimators: int = 100) -> AnomalyDetector:
    """Factory function to get anomaly detector instance"""
    return AnomalyDetector(contamination=contamination, n_estimators=n_estimators)


if __name__ == "__main__":
    # Test with sample data
    print("=" * 50)
    print("ANOMALY DETECTION TEST")
    print("=" * 50)
    
    # Create sample data
    sample_data = pd.DataFrame({
        'event_id': ['EVT001', 'EVT002', 'EVT003', 'EVT004', 'EVT005'],
        'timestamp': ['2024-11-01 10:00:00', '2024-11-01 03:00:00', '2024-11-01 14:00:00', 
                      '2024-11-01 11:00:00', '2024-11-01 02:30:00'],
        'user_id': ['USR001', 'USR001', 'USR002', 'USR002', 'USR003'],
        'event_type': ['login', 'database_access', 'file_access', 'login', 'data_export'],
        'location': ['Mumbai', 'Beijing', 'Delhi', 'Delhi', 'Unknown'],
        'access_level': ['Level-1', 'Level-3', 'Level-2', 'Level-2', 'Level-3'],
        'transaction_amount': [None, None, 5000, None, 9000000],
        'ip_address': ['192.168.1.10', '45.67.89.100', '192.168.2.20', '192.168.2.21', '10.0.0.1'],
        'duration_mins': [5, 120, 30, 10, 180],
        'device_id': ['DEV_A01', 'DEV_A15', 'DEV_A02', 'DEV_A02', 'DEV_UNKNOWN'],
        'status': ['success', 'success', 'success', 'success', 'success'],
        'is_anomaly': [0, 1, 0, 0, 1]
    })
    
    detector = AnomalyDetector(contamination=0.4, n_estimators=50)
    results = detector.fit_predict(sample_data)
    
    print("\nPredicted Results:")
    print(results[['event_id', 'timestamp', 'is_anomaly', 'predicted_anomaly', 'risk_score']])