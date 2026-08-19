#!/usr/bin/env python3
"""
SmartPay Salary Prediction API & Inference Module

Usage:
    As a library:
        from salary_predictor import SalaryPredictor
        predictor = SalaryPredictor('path/to/model.pkl')
        salary = predictor.predict({...})
    
    As a service:
        python salary_predictor.py
        # Opens Flask API at http://localhost:5000
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Union
from datetime import datetime
import numpy as np
import pandas as pd
import joblib

# Optional Flask import for API
try:
    from flask import Flask, request, jsonify
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SalaryPredictor:
    """Production-ready salary prediction interface."""
    
    def __init__(self, model_path: Union[str, Path], metrics_path: Union[str, Path] = None):
        """
        Initialize predictor with a trained pipeline.
        
        Args:
            model_path: Path to saved pipeline (pkl format)
            metrics_path: Optional path to metrics JSON for reference
        """
        self.model_path = Path(model_path)
        self.metrics_path = Path(metrics_path) if metrics_path else None
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        # Load pipeline
        logger.info(f"Loading model from {self.model_path}")
        self.pipeline = joblib.load(self.model_path)
        
        # Load metrics if available
        self.metrics = {}
        if self.metrics_path and self.metrics_path.exists():
            with open(self.metrics_path, 'r') as f:
                self.metrics = json.load(f)
            logger.info(f"Loaded model metrics: R²={self.metrics.get('test_metrics', {}).get('R2', 'N/A')}")
        
        self.prediction_count = 0
        logger.info("Predictor initialized successfully")
    
    def predict(self, employee_data: Dict[str, Any]) -> float:
        """
        Predict salary for an employee.
        
        Args:
            employee_data: Dictionary with employee features
                Expected keys: job_title, experience_years, education_level,
                              skills_count, industry, company_size, location,
                              remote_work, certifications
        
        Returns:
            Predicted salary as float
            
        Raises:
            ValueError: If required fields are missing
        """
        # Validate input
        required_fields = [
            'job_title', 'experience_years', 'education_level', 'skills_count',
            'industry', 'company_size', 'location', 'remote_work', 'certifications'
        ]
        
        missing_fields = [f for f in required_fields if f not in employee_data]
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
        
        # Create DataFrame
        df = pd.DataFrame([employee_data])
        
        # Make prediction
        try:
            prediction = float(self.pipeline.predict(df)[0])
            self.prediction_count += 1
            
            logger.info(
                f"Prediction #{self.prediction_count}: {employee_data.get('job_title')} "
                f"with {employee_data.get('experience_years')} years → ₹{prediction:,.2f}"
            )
            
            return prediction
        
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise
    
    def predict_batch(self, employees_data: list) -> list:
        """
        Predict salaries for multiple employees.
        
        Args:
            employees_data: List of employee dictionaries
            
        Returns:
            List of predicted salaries
        """
        logger.info(f"Processing batch prediction for {len(employees_data)} employees")
        
        predictions = []
        for i, employee in enumerate(employees_data):
            try:
                pred = self.predict(employee)
                predictions.append({'prediction': pred, 'success': True})
            except Exception as e:
                logger.warning(f"Failed to predict for employee {i}: {str(e)}")
                predictions.append({'error': str(e), 'success': False})
        
        return predictions
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            'model_path': str(self.model_path),
            'model_loaded': True,
            'metrics': self.metrics,
            'predictions_made': self.prediction_count,
            'timestamp': datetime.now().isoformat()
        }


def create_flask_app(model_path: str) -> Flask:
    """
    Create Flask API application.
    
    Args:
        model_path: Path to trained model
        
    Returns:
        Flask application instance
    """
    app = Flask(__name__)
    
    # Initialize predictor
    try:
        predictor = SalaryPredictor(model_path)
        logger.info("Flask app initialized with predictor")
    except Exception as e:
        logger.error(f"Failed to initialize predictor: {e}")
        raise
    
    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint."""
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'model_loaded': True
        }), 200
    
    @app.route('/info', methods=['GET'])
    def model_info():
        """Get model information."""
        return jsonify(predictor.get_model_info()), 200
    
    @app.route('/predict', methods=['POST'])
    def predict():
        """
        Predict salary endpoint.
        
        Expected JSON:
        {
            "job_title": "Software Engineer",
            "experience_years": 7,
            "education_level": "Bachelor",
            "skills_count": 12,
            "industry": "Technology",
            "company_size": "Medium",
            "location": "India",
            "remote_work": "Hybrid",
            "certifications": 3
        }
        """
        try:
            data = request.get_json()
            
            if not data:
                return jsonify({'error': 'No JSON data provided'}), 400
            
            prediction = predictor.predict(data)
            
            return jsonify({
                'prediction': round(prediction, 2),
                'currency': 'INR',
                'timestamp': datetime.now().isoformat(),
                'input': data
            }), 200
        
        except ValueError as e:
            return jsonify({'error': str(e)}), 400
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return jsonify({'error': 'Internal server error'}), 500
    
    @app.route('/predict_batch', methods=['POST'])
    def predict_batch():
        """
        Batch prediction endpoint.
        
        Expected JSON:
        {
            "employees": [
                {...},
                {...}
            ]
        }
        """
        try:
            data = request.get_json()
            
            if not data or 'employees' not in data:
                return jsonify({'error': 'No employees data provided'}), 400
            
            predictions = predictor.predict_batch(data['employees'])
            
            return jsonify({
                'predictions': predictions,
                'count': len(predictions),
                'successful': sum(1 for p in predictions if p.get('success', False)),
                'timestamp': datetime.now().isoformat()
            }), 200
        
        except Exception as e:
            logger.error(f"Batch prediction error: {str(e)}")
            return jsonify({'error': 'Internal server error'}), 500
    
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'error': 'Endpoint not found'}), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({'error': 'Internal server error'}), 500
    
    return app


def main():
    """Run the Flask API server."""
    if not FLASK_AVAILABLE:
        print("Flask is required to run the API server.")
        print("Install it with: pip install flask")
        return
    
    # Path to model
    MODEL_PATH = Path.cwd() / 'smartpay_project' / 'models' / 'best_salary_regressor.pkl'
    
    if not MODEL_PATH.exists():
        print(f"Model not found at {MODEL_PATH}")
        print("Please train the model first by running the Jupyter notebook.")
        return
    
    # Create and run app
    app = create_flask_app(str(MODEL_PATH))
    
    print("\n" + "="*60)
    print("SmartPay Salary Prediction API".center(60))
    print("="*60)
    print("\nEndpoints:")
    print("  GET  /health          - Health check")
    print("  GET  /info            - Model information")
    print("  POST /predict         - Single prediction")
    print("  POST /predict_batch   - Batch predictions")
    print("\nStarting server at http://localhost:5000")
    print("="*60 + "\n")
    
    # Run with development server (use gunicorn for production)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)


if __name__ == '__main__':
    main()
