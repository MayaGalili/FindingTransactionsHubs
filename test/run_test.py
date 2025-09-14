#!/usr/bin/env python3
"""
Simple test runner for CTG analysis

This script runs the CTG analysis on the test dataset and shows the results.
"""

import os
import sys
import subprocess

def main():
    """Run the CTG analysis test"""
    print("🧪 Running CTG Analysis Test")
    print("="*50)
    
    # Check if test data exists
    test_data_path = "resources/input.tsv"
    if not os.path.exists(test_data_path):
        print(f"❌ Test data not found at {test_data_path}")
        print("Please make sure the input.tsv file exists in test/resources/")
        return False
    
    # Run the test
    try:
        print(f"📊 Test data: {test_data_path}")
        print("🚀 Starting analysis...")
        
        # Run the test using uv
        result = subprocess.run([
            "uv", "run", "python", "test/test_ctg_analysis.py"
        ], capture_output=True, text=True)
        
        print("📋 Test Output:")
        print("-" * 50)
        print(result.stdout)
        
        if result.stderr:
            print("⚠️  Warnings/Errors:")
            print("-" * 50)
            print(result.stderr)
        
        if result.returncode == 0:
            print("✅ Test completed successfully!")
            return True
        else:
            print(f"❌ Test failed with return code: {result.returncode}")
            return False
            
    except FileNotFoundError:
        print("❌ 'uv' command not found. Please install uv first:")
        print("   curl -LsSf https://astral.sh/uv/install.sh | sh")
        return False
    except Exception as e:
        print(f"❌ Error running test: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
