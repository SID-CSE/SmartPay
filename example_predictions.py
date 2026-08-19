#!/usr/bin/env python3
"""
SmartPay Example Predictions

This script demonstrates how to use the trained salary prediction model
to make predictions for various employee profiles.

Run after training the model with the Jupyter notebook.
"""

from pathlib import Path
from salary_predictor import SalaryPredictor
import json


def main():
    """Run example predictions."""
    
    # Path to the trained model
    model_path = Path('smartpay_project/models/best_salary_regressor.pkl')
    
    if not model_path.exists():
        print("❌ Model not found!")
        print(f"   Expected at: {model_path}")
        print("   Please run the Jupyter notebook first to train the model.")
        return
    
    print("=" * 70)
    print("SmartPay Salary Prediction - Examples".center(70))
    print("=" * 70)
    print()
    
    # Initialize predictor
    try:
        predictor = SalaryPredictor(model_path)
        print("✓ Model loaded successfully")
        print()
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Define example employees
    examples = {
        "Junior Web Developer": {
            "job_title": "Junior Web Developer",
            "experience_years": 2,
            "education_level": "Bachelor",
            "skills_count": 6,
            "industry": "Technology",
            "company_size": "Small",
            "location": "India",
            "remote_work": "Yes",
            "certifications": 1
        },
        "Mid-level Software Engineer": {
            "job_title": "Software Engineer",
            "experience_years": 7,
            "education_level": "Bachelor",
            "skills_count": 12,
            "industry": "Technology",
            "company_size": "Medium",
            "location": "India",
            "remote_work": "Hybrid",
            "certifications": 3
        },
        "Senior Data Scientist": {
            "job_title": "Data Scientist",
            "experience_years": 12,
            "education_level": "Master",
            "skills_count": 18,
            "industry": "Finance",
            "company_size": "Large",
            "location": "USA",
            "remote_work": "Yes",
            "certifications": 5
        },
        "ML Engineer (Startup)": {
            "job_title": "Machine Learning Engineer",
            "experience_years": 5,
            "education_level": "Master",
            "skills_count": 14,
            "industry": "Technology",
            "company_size": "Small",
            "location": "India",
            "remote_work": "Yes",
            "certifications": 2
        },
        "Product Manager (Tech)": {
            "job_title": "Product Manager",
            "experience_years": 8,
            "education_level": "Bachelor",
            "skills_count": 10,
            "industry": "Technology",
            "company_size": "Large",
            "location": "USA",
            "remote_work": "Hybrid",
            "certifications": 1
        },
        "DevOps Engineer": {
            "job_title": "DevOps Engineer",
            "experience_years": 6,
            "education_level": "Bachelor",
            "skills_count": 11,
            "industry": "Technology",
            "company_size": "Medium",
            "location": "India",
            "remote_work": "No",
            "certifications": 4
        },
        "Finance Analyst": {
            "job_title": "Finance Analyst",
            "experience_years": 4,
            "education_level": "Bachelor",
            "skills_count": 8,
            "industry": "Finance",
            "company_size": "Large",
            "location": "USA",
            "remote_work": "No",
            "certifications": 2
        },
        "Lead Architect": {
            "job_title": "Lead Architect",
            "experience_years": 15,
            "education_level": "Master",
            "skills_count": 20,
            "industry": "Technology",
            "company_size": "Enterprise",
            "location": "USA",
            "remote_work": "Hybrid",
            "certifications": 6
        }
    }
    
    # Make predictions for all examples
    predictions = []
    
    print("SINGLE PREDICTIONS:")
    print("-" * 70)
    
    for profile_name, profile_data in examples.items():
        try:
            salary = predictor.predict(profile_data)
            predictions.append({
                'profile': profile_name,
                'salary': salary,
                'data': profile_data
            })
            
            # Print result
            print(f"\n📊 {profile_name}")
            print(f"   Role: {profile_data['job_title']}")
            print(f"   Experience: {profile_data['experience_years']} years")
            print(f"   Location: {profile_data['location']}")
            print(f"   Company Size: {profile_data['company_size']}")
            print(f"   Education: {profile_data['education_level']}")
            print(f"   Predicted Salary: ₹{salary:,.2f}")
            
        except Exception as e:
            print(f"\n❌ Error predicting for {profile_name}: {e}")
    
    # Batch prediction example
    print("\n" + "=" * 70)
    print("BATCH PREDICTION EXAMPLE:")
    print("-" * 70)
    print()
    
    batch_employees = [
        examples["Junior Web Developer"],
        examples["Mid-level Software Engineer"],
        examples["Senior Data Scientist"]
    ]
    
    try:
        batch_results = predictor.predict_batch(batch_employees)
        
        print(f"Batch Processing: {len(batch_employees)} employees")
        print()
        
        for i, (employee, result) in enumerate(zip(batch_employees, batch_results), 1):
            if result.get('success', False):
                print(f"{i}. {employee['job_title']} → ₹{result['prediction']:,.2f}")
            else:
                print(f"{i}. {employee['job_title']} → ERROR: {result.get('error', 'Unknown')}")
    
    except Exception as e:
        print(f"❌ Batch prediction failed: {e}")
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS:")
    print("-" * 70)
    print()
    
    if predictions:
        salaries = [p['salary'] for p in predictions]
        avg_salary = sum(salaries) / len(salaries)
        min_salary = min(salaries)
        max_salary = max(salaries)
        salary_range = max_salary - min_salary
        
        print(f"Total Profiles Analyzed: {len(predictions)}")
        print(f"Average Predicted Salary: ₹{avg_salary:,.2f}")
        print(f"Minimum Salary: ₹{min_salary:,.2f}")
        print(f"Maximum Salary: ₹{max_salary:,.2f}")
        print(f"Salary Range: ₹{salary_range:,.2f}")
        print()
        
        # Salary by category
        print("Salary by Experience Level:")
        by_experience = {}
        for pred in predictions:
            years = pred['data']['experience_years']
            if years not in by_experience:
                by_experience[years] = []
            by_experience[years].append(pred['salary'])
        
        for years in sorted(by_experience.keys()):
            avg = sum(by_experience[years]) / len(by_experience[years])
            print(f"  {years} years: ₹{avg:,.2f}")
    
    # Save results to JSON
    print()
    print("=" * 70)
    print("SAVING RESULTS:")
    print("-" * 70)
    print()
    
    output_file = Path('smartpay_project/results/example_predictions.json')
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump({
            'predictions': predictions,
            'summary': {
                'total_profiles': len(predictions),
                'average_salary': sum(p['salary'] for p in predictions) / len(predictions) if predictions else 0,
                'min_salary': min(p['salary'] for p in predictions) if predictions else 0,
                'max_salary': max(p['salary'] for p in predictions) if predictions else 0,
            }
        }, f, indent=2)
    
    print(f"✓ Results saved to: {output_file}")
    print()
    print("=" * 70)
    print("Predictions Complete! 🎉".center(70))
    print("=" * 70)


if __name__ == '__main__':
    main()
