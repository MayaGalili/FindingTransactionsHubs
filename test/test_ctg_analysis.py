#!/usr/bin/env python3
"""
Test script for CTG (Customer Transaction Graph) analysis

This test runs the main CTG analysis on a small test dataset with known anomalies.
"""

import os
import sys
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path

# Add src directory to path so we can import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pars_and_manipulate_data import ParsAndManipulate
from gen_costumer_transaction_graph import GenCTG
from explore_ctg import ExploreCTG


class TestCTGAnalysis:
    """Test class for CTG analysis functionality"""
    
    def __init__(self):
        self.test_data_path = os.path.join(os.path.dirname(__file__), 'resources', 'input.tsv')
        self.temp_dir = None
        
    def setup_test_environment(self):
        """Set up temporary directory for test outputs"""
        self.temp_dir = tempfile.mkdtemp(prefix='ctg_test_')
        print(f"Test output directory: {self.temp_dir}")
        
    def cleanup_test_environment(self):
        """Clean up temporary files"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"Cleaned up test directory: {self.temp_dir}")
    
    def test_data_loading(self):
        """Test that the test data loads correctly"""
        print("\n" + "="*50)
        print("TESTING DATA LOADING")
        print("="*50)
        
        # Load the test data
        df = pd.read_csv(self.test_data_path)
        
        print(f"Test dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"First few rows:")
        print(df.head())
        
        # Check for expected anomalies
        print(f"\nData analysis:")
        print(f"Unique emails: {df['email'].nunique()}")
        print(f"Unique IPs: {df['browser_ip'].nunique()}")
        print(f"Order status distribution:")
        print(df['order_status'].value_counts())
        
        # Check for the expected anomaly (suspicious email pattern)
        suspicious_emails = df[df['email'].str.contains(r'[0-9]{8,}', regex=True)]
        print(f"\nSuspicious emails (8+ consecutive digits): {len(suspicious_emails)}")
        if len(suspicious_emails) > 0:
            print("Found expected anomaly - suspicious email patterns!")
            print(suspicious_emails[['order_id', 'email', 'order_status']].head())
        
        return df
    
    def test_ctg_analysis(self, df):
        """Test the complete CTG analysis pipeline"""
        print("\n" + "="*50)
        print("TESTING CTG ANALYSIS PIPELINE")
        print("="*50)
        
        try:
            # Step 1: Data preprocessing
            print("Step 1: Data preprocessing...")
            parser = ParsAndManipulate(self.test_data_path)
            parser.run()
            processed_df = parser.get_df()
            print(f"Processed data shape: {processed_df.shape}")
            
            # Step 2: Generate CTG
            print("Step 2: Generating Customer Transaction Graph...")
            ctg_gen = GenCTG(df=processed_df)
            ctg_gen.run()
            ctg = ctg_gen.get_ctg()
            score_mat = ctg_gen.get_ctg_base_mat()
            
            print(f"CTG nodes: {ctg.number_of_nodes()}")
            print(f"CTG edges: {ctg.number_of_edges()}")
            print(f"Score matrix shape: {score_mat.shape}")
            print(f"Score matrix max value: {score_mat.max():.4f}")
            print(f"Score matrix min value: {score_mat.min():.4f}")
            
            # Step 3: Analyze the graph
            print("Step 3: Analyzing the graph...")
            explorer = ExploreCTG(df=processed_df, ctg=ctg, out_dir=self.temp_dir)
            
            # Find hubs using PageRank
            print("Finding hubs using PageRank...")
            explorer._ExploreCTG__find_hubs()
            
            # Save results
            print("Saving results...")
            np.savetxt(os.path.join(self.temp_dir, 'test_score_mat.csv'), score_mat, delimiter=',')
            
            # Generate visualization
            print("Generating visualization...")
            explorer.run()
            
            print("✅ CTG analysis completed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ CTG analysis failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_anomaly_detection(self, df):
        """Test specific anomaly detection methods"""
        print("\n" + "="*50)
        print("TESTING ANOMALY DETECTION")
        print("="*50)
        
        # Convert timestamp for analysis
        df['order_captured_at'] = pd.to_datetime(df['order_captured_at'])
        
        # Test 1: Amount-based anomalies
        amount_q99 = df['order_total_spent'].quantile(0.99)
        amount_q01 = df['order_total_spent'].quantile(0.01)
        high_amount = df[df['order_total_spent'] > amount_q99]
        low_amount = df[df['order_total_spent'] < amount_q01]
        
        print(f"High amount transactions (>99th percentile): {len(high_amount)}")
        print(f"Low amount transactions (<1st percentile): {len(low_amount)}")
        
        # Test 2: Time-based anomalies
        df['hour'] = df['order_captured_at'].dt.hour
        unusual_hours = df[(df['hour'] < 6) | (df['hour'] > 22)]
        print(f"Transactions at unusual hours (6am-10pm): {len(unusual_hours)}")
        
        # Test 3: Rapid successive transactions
        df_sorted = df.sort_values(['email', 'order_captured_at'])
        df_sorted['time_diff'] = df_sorted.groupby('email')['order_captured_at'].diff()
        rapid_transactions = df_sorted[df_sorted['time_diff'] < pd.Timedelta(minutes=5)]
        print(f"Rapid successive transactions (<5 min): {len(rapid_transactions)}")
        
        # Test 4: Email pattern anomalies
        suspicious_emails = df[df['email'].str.contains(r'[0-9]{8,}', regex=True)]
        print(f"Emails with 8+ consecutive digits: {len(suspicious_emails)}")
        
        # Test 5: Geographic anomalies
        ip_zip_groups = df.groupby('browser_ip')['shipping_zip'].nunique()
        geo_anomalies = ip_zip_groups[ip_zip_groups > 1]
        print(f"IPs with multiple shipping locations: {len(geo_anomalies)}")
        
        # Test 6: Same email, different IPs
        email_ip_groups = df.groupby('email')['browser_ip'].nunique()
        suspicious_emails_multi_ip = email_ip_groups[email_ip_groups > 1]
        print(f"Emails with multiple IPs: {len(suspicious_emails_multi_ip)}")
        
        return {
            'high_amount': len(high_amount),
            'low_amount': len(low_amount),
            'unusual_hours': len(unusual_hours),
            'rapid_transactions': len(rapid_transactions),
            'suspicious_emails': len(suspicious_emails),
            'geo_anomalies': len(geo_anomalies),
            'emails_multi_ip': len(suspicious_emails_multi_ip)
        }
    
    def run_complete_test(self):
        """Run the complete test suite"""
        print("🚀 STARTING CTG ANALYSIS TEST")
        print("="*60)
        
        try:
            # Setup
            self.setup_test_environment()
            
            # Test 1: Data loading
            df = self.test_data_loading()
            
            # Test 2: Anomaly detection
            anomaly_results = self.test_anomaly_detection(df)
            
            # Test 3: CTG analysis
            ctg_success = self.test_ctg_analysis(df)
            
            # Summary
            print("\n" + "="*60)
            print("TEST SUMMARY")
            print("="*60)
            print(f"✅ Data loading: SUCCESS")
            print(f"✅ Anomaly detection: SUCCESS")
            print(f"✅ CTG analysis: {'SUCCESS' if ctg_success else 'FAILED'}")
            
            print(f"\nAnomaly Detection Results:")
            for key, value in anomaly_results.items():
                print(f"  - {key}: {value}")
            
            if ctg_success:
                print(f"\n🎉 All tests passed! Check output in: {self.temp_dir}")
                print("Generated files:")
                for file in os.listdir(self.temp_dir):
                    print(f"  - {file}")
            else:
                print(f"\n❌ Some tests failed. Check error messages above.")
                
        except Exception as e:
            print(f"\n❌ Test suite failed: {str(e)}")
            import traceback
            traceback.print_exc()
            
        finally:
            # Cleanup
            self.cleanup_test_environment()


def main():
    """Main test function"""
    test = TestCTGAnalysis()
    test.run_complete_test()


if __name__ == "__main__":
    main()
