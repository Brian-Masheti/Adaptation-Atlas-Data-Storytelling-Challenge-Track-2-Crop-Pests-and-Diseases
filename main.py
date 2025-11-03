#!/usr/bin/env python3
"""
Climate-Driven Agricultural Risk Analysis Framework
Main Entry Point

Author: Brian Savatia Masheti
Role: Data Analyst and Developer
Competition: Zindi Adaptation Atlas Challenge - Track 2
Project: Agricultural risk analysis framework
Date: November 2024
"""

import sys
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from climate_risk_analyzer import ClimateRiskAnalyzer

def main():
    """
    Main execution function for the Climate Risk Analysis Framework.
    
    This function demonstrates the complete workflow for agricultural
    risk analysis and climate adaptation planning.
    """
    
    print("🌍 Climate-Driven Agricultural Risk Analysis Framework")
    print("=" * 60)
    print("🌍 Zindi Adaptation Atlas Challenge - Track 2")
    print("🎯 Agricultural Risk Analysis and Climate Adaptation")
    print("=" * 60)
    
    try:
        # Initialize the Climate Risk Analyzer
        print("🚀 Initializing Climate Risk Analyzer...")
        analyzer = ClimateRiskAnalyzer()
        
        # Create demonstration data (in real usage, load actual data)
        print("📊 Loading and preparing data...")
        X_train, X_test, y_train, y_test = analyzer.create_winning_data()
        
        # Train the ensemble models
        print("🧠 Training ensemble models...")
        analyzer.train_ensemble(X_train, y_train)
        
        # Make predictions
        print("🔮 Making predictions...")
        predictions = analyzer.predict(X_test)
        
        # Evaluate performance
        print("📈 Evaluating performance...")
        score = analyzer.evaluate_performance(X_test, y_test)
        
        print(f"\n🎉 Analysis Complete!")
        print(f"📊 Final Score: {score:.3f}")
        print(f"✅ Agricultural Risk Analysis Framework executed successfully!")
        
        # Save results
        analyzer.save_results(predictions, "outputs/predictions.csv")
        print("💾 Results saved to outputs/predictions.csv")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during execution: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
