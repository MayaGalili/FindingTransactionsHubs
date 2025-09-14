#!/usr/bin/env python3
"""
Transaction Anomaly Detection Analysis Script

This script analyzes the transaction dataset to understand:
1. Data structure and quality
2. Current CTG (Customer Transaction Graph) approach
3. Potential anomalies and patterns
4. Recommendations for improvement
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

def load_and_explore_data():
    """Load and perform initial data exploration"""
    print("=" * 60)
    print("LOADING AND EXPLORING TRANSACTION DATA")
    print("=" * 60)
    
    # Load the dataset
    df = pd.read_csv('data/customer_trust_dataset_full.csv')
    
    print(f"Dataset shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print("\nColumn names and types:")
    print(df.dtypes)
    
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nBasic statistics for numerical columns:")
    print(df.describe())
    
    print("\nMissing values per column:")
    missing_data = df.isnull().sum()
    print(missing_data[missing_data > 0])
    
    print("\nOrder status distribution:")
    print(df['order_status'].value_counts())
    
    print("\nUnique values in key columns:")
    print(f"Unique emails: {df['email'].nunique()}")
    print(f"Unique IPs: {df['browser_ip'].nunique()}")
    print(f"Unique shipping addresses: {df['shipping_address1'].nunique()}")
    print(f"Unique order IDs: {df['order_id'].nunique()}")
    
    return df

def analyze_transaction_patterns(df):
    """Analyze patterns that might indicate anomalies"""
    print("\n" + "=" * 60)
    print("ANALYZING TRANSACTION PATTERNS")
    print("=" * 60)
    
    # Amount analysis
    print("Transaction amount analysis:")
    print(f"Min amount: ${df['order_total_spent'].min()}")
    print(f"Max amount: ${df['order_total_spent'].max()}")
    print(f"Mean amount: ${df['order_total_spent'].mean():.2f}")
    print(f"Median amount: ${df['order_total_spent'].median():.2f}")
    print(f"Std amount: ${df['order_total_spent'].std():.2f}")
    
    # Time analysis
    df['order_captured_at'] = pd.to_datetime(df['order_captured_at'])
    print(f"\nTime range: {df['order_captured_at'].min()} to {df['order_captured_at'].max()}")
    print(f"Duration: {(df['order_captured_at'].max() - df['order_captured_at'].min()).total_seconds() / 3600:.1f} hours")
    
    # Duplicate analysis
    print("\nDuplicate analysis:")
    print(f"Duplicate emails: {df['email'].duplicated().sum()}")
    print(f"Duplicate IPs: {df['browser_ip'].duplicated().sum()}")
    print(f"Duplicate addresses: {df['shipping_address1'].duplicated().sum()}")
    
    # Suspicious patterns
    print("\nSuspicious patterns:")
    
    # Same email, different IPs
    email_ip_groups = df.groupby('email')['browser_ip'].nunique()
    suspicious_emails = email_ip_groups[email_ip_groups > 1]
    print(f"Emails with multiple IPs: {len(suspicious_emails)}")
    
    # Same IP, different emails
    ip_email_groups = df.groupby('browser_ip')['email'].nunique()
    suspicious_ips = ip_email_groups[ip_email_groups > 1]
    print(f"IPs with multiple emails: {len(suspicious_ips)}")
    
    # Same address, different emails
    addr_email_groups = df.groupby('shipping_address1')['email'].nunique()
    suspicious_addresses = addr_email_groups[addr_email_groups > 1]
    print(f"Addresses with multiple emails: {len(suspicious_addresses)}")
    
    return {
        'suspicious_emails': suspicious_emails,
        'suspicious_ips': suspicious_ips,
        'suspicious_addresses': suspicious_addresses
    }

def analyze_current_ctg_approach(df):
    """Analyze the current Customer Transaction Graph approach"""
    print("\n" + "=" * 60)
    print("ANALYZING CURRENT CTG APPROACH")
    print("=" * 60)
    
    # Simulate the current approach
    print("Current approach creates a Customer Transaction Graph (CTG) by:")
    print("1. Converting categorical features to integers")
    print("2. Creating a similarity matrix between transactions")
    print("3. Using top 0.1% most similar transactions to build a graph")
    print("4. Analyzing the graph for hubs and communities")
    
    # Show the features used
    selected_features = [
        'order_total_spent', 'browser_ip', 'email', 'shipping_zip',
        'normalized_full_shipping_name', 'order_captured_at', 
        'shipping_address1', 'order_status'
    ]
    
    print(f"\nFeatures used for similarity: {selected_features}")
    
    # Analyze the approach
    print("\nStrengths of current approach:")
    print("- Uses multiple features for similarity")
    print("- Graph-based analysis can find complex patterns")
    print("- PageRank can identify important/hub transactions")
    
    print("\nLimitations of current approach:")
    print("- Only uses top 0.1% similarity (might miss subtle patterns)")
    print("- No explicit anomaly scoring")
    print("- No consideration of temporal patterns")
    print("- No handling of missing values")
    print("- No feature engineering for anomaly detection")

def identify_anomaly_indicators(df):
    """Identify potential anomaly indicators in the data"""
    print("\n" + "=" * 60)
    print("IDENTIFYING ANOMALY INDICATORS")
    print("=" * 60)
    
    anomalies = []
    
    # 1. Amount-based anomalies
    amount_q99 = df['order_total_spent'].quantile(0.99)
    amount_q01 = df['order_total_spent'].quantile(0.01)
    high_amount = df[df['order_total_spent'] > amount_q99]
    low_amount = df[df['order_total_spent'] < amount_q01]
    
    print(f"High amount transactions (>99th percentile): {len(high_amount)}")
    print(f"Low amount transactions (<1st percentile): {len(low_amount)}")
    
    # 2. Time-based anomalies
    df['hour'] = df['order_captured_at'].dt.hour
    df['minute'] = df['order_captured_at'].dt.minute
    
    # Transactions at unusual hours
    unusual_hours = df[(df['hour'] < 6) | (df['hour'] > 22)]
    print(f"Transactions at unusual hours (6am-10pm): {len(unusual_hours)}")
    
    # 3. Pattern-based anomalies
    # Same email, multiple transactions in short time
    df_sorted = df.sort_values(['email', 'order_captured_at'])
    df_sorted['time_diff'] = df_sorted.groupby('email')['order_captured_at'].diff()
    rapid_transactions = df_sorted[df_sorted['time_diff'] < pd.Timedelta(minutes=5)]
    print(f"Rapid successive transactions (<5 min): {len(rapid_transactions)}")
    
    # 4. Geographic anomalies
    # Same IP, different locations
    ip_zip_groups = df.groupby('browser_ip')['shipping_zip'].nunique()
    geo_anomalies = ip_zip_groups[ip_zip_groups > 1]
    print(f"IPs with multiple shipping locations: {len(geo_anomalies)}")
    
    # 5. Email pattern anomalies
    # Suspicious email patterns
    suspicious_emails = df[df['email'].str.contains(r'[0-9]{8,}', regex=True)]
    print(f"Emails with 8+ consecutive digits: {len(suspicious_emails)}")
    
    return {
        'high_amount': high_amount,
        'low_amount': low_amount,
        'unusual_hours': unusual_hours,
        'rapid_transactions': rapid_transactions,
        'geo_anomalies': geo_anomalies,
        'suspicious_emails': suspicious_emails
    }

def recommend_improvements():
    """Provide recommendations for improving anomaly detection"""
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS FOR IMPROVEMENT")
    print("=" * 60)
    
    print("1. FEATURE ENGINEERING:")
    print("   - Create time-based features (hour, day of week, time since last transaction)")
    print("   - Add email pattern features (length, special chars, domain)")
    print("   - Create geographic features (IP geolocation, distance between IP and shipping)")
    print("   - Add behavioral features (transaction frequency, amount patterns)")
    
    print("\n2. ANOMALY DETECTION METHODS:")
    print("   - Isolation Forest for unsupervised anomaly detection")
    print("   - Local Outlier Factor (LOF) for density-based anomalies")
    print("   - One-Class SVM for complex pattern detection")
    print("   - Statistical methods (Z-score, IQR) for simple anomalies")
    
    print("\n3. GRAPH-BASED IMPROVEMENTS:")
    print("   - Use different similarity thresholds (not just top 0.1%)")
    print("   - Add edge weights based on feature importance")
    print("   - Use community detection algorithms")
    print("   - Implement graph neural networks")
    
    print("\n4. ENSEMBLE APPROACH:")
    print("   - Combine multiple anomaly detection methods")
    print("   - Use voting or weighted scoring")
    print("   - Implement different thresholds for different anomaly types")
    
    print("\n5. EVALUATION:")
    print("   - Create labeled test set with known anomalies")
    print("   - Use precision, recall, F1-score for evaluation")
    print("   - Implement cross-validation")
    print("   - Create confusion matrix for different anomaly types")

def create_visualizations(df, anomalies):
    """Create visualizations to understand the data better"""
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Transaction Data Analysis', fontsize=16)
    
    # 1. Amount distribution
    axes[0, 0].hist(df['order_total_spent'], bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('Transaction Amount Distribution')
    axes[0, 0].set_xlabel('Amount ($)')
    axes[0, 0].set_ylabel('Frequency')
    
    # 2. Order status distribution
    status_counts = df['order_status'].value_counts()
    axes[0, 1].pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%')
    axes[0, 1].set_title('Order Status Distribution')
    
    # 3. Hourly transaction pattern
    df['hour'] = df['order_captured_at'].dt.hour
    hourly_counts = df['hour'].value_counts().sort_index()
    axes[0, 2].bar(hourly_counts.index, hourly_counts.values)
    axes[0, 2].set_title('Transactions by Hour')
    axes[0, 2].set_xlabel('Hour of Day')
    axes[0, 2].set_ylabel('Number of Transactions')
    
    # 4. Amount vs Status
    df.boxplot(column='order_total_spent', by='order_status', ax=axes[1, 0])
    axes[1, 0].set_title('Amount Distribution by Status')
    axes[1, 0].set_xlabel('Order Status')
    axes[1, 0].set_ylabel('Amount ($)')
    
    # 5. IP frequency (top 20)
    ip_counts = df['browser_ip'].value_counts().head(20)
    axes[1, 1].barh(range(len(ip_counts)), ip_counts.values)
    axes[1, 1].set_yticks(range(len(ip_counts)))
    axes[1, 1].set_yticklabels(ip_counts.index, fontsize=8)
    axes[1, 1].set_title('Top 20 IPs by Transaction Count')
    axes[1, 1].set_xlabel('Number of Transactions')
    
    # 6. Email frequency (top 20)
    email_counts = df['email'].value_counts().head(20)
    axes[1, 2].barh(range(len(email_counts)), email_counts.values)
    axes[1, 2].set_yticks(range(len(email_counts)))
    axes[1, 2].set_yticklabels(email_counts.index, fontsize=8)
    axes[1, 2].set_title('Top 20 Emails by Transaction Count')
    axes[1, 2].set_xlabel('Number of Transactions')
    
    plt.tight_layout()
    plt.savefig('transaction_analysis.png', dpi=300, bbox_inches='tight')
    print("Visualizations saved as 'transaction_analysis.png'")
    
    return fig

def main():
    """Main analysis function"""
    print("TRANSACTION ANOMALY DETECTION ANALYSIS")
    print("=" * 60)
    
    # Load and explore data
    df = load_and_explore_data()
    
    # Analyze patterns
    patterns = analyze_transaction_patterns(df)
    
    # Analyze current approach
    analyze_current_ctg_approach(df)
    
    # Identify anomalies
    anomalies = identify_anomaly_indicators(df)
    
    # Create visualizations
    create_visualizations(df, anomalies)
    
    # Provide recommendations
    recommend_improvements()
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print("Next steps:")
    print("1. Review the generated visualizations")
    print("2. Implement recommended improvements")
    print("3. Test different anomaly detection methods")
    print("4. Evaluate performance on test data")

if __name__ == "__main__":
    main()

