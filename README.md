# Finding Transactions Hubs

A data science project for detecting abnormal transactions using Customer Transaction Graph (CTG) approach and advanced anomaly detection techniques.

## 🎯 Purpose

This project analyzes transaction datasets to identify fraudulent or suspicious activities by:
- Building Customer Transaction Graphs (CTG) based on transaction similarities
- Detecting anomalies using multiple machine learning approaches
- Providing comprehensive analysis and visualization tools
- Offering both graph-based and statistical anomaly detection methods

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd FindingTransactionsHubs
   ```

2. **Install dependencies using uv**
   ```bash
   uv sync
   ```

3. **Run the analysis**
   ```bash
   # Run comprehensive analysis script
   uv run python analyze_transaction_data.py
   
   # Or run the interactive Jupyter notebook
   uv run jupyter notebook transaction_anomaly_analysis.ipynb
   
   # Or run the original CTG analysis
   uv run python src/run_ctg_analysis.py
   ```

## 📊 Data

The project works with transaction datasets containing:
- Order information (ID, amount, timestamp, status)
- Customer details (email, IP address, shipping info)
- Payment data (credit card BIN, browser info)

**Sample data files:**
- `data/customer_trust_dataset_full.csv` - Full dataset (4,000 transactions)
- `data/customer_trust_dataset_test.csv` - Test dataset

## 🔍 Current Approach

### Customer Transaction Graph (CTG)
1. **Feature Engineering**: Converts categorical features to numerical values
2. **Similarity Matrix**: Calculates similarity between all transaction pairs
3. **Graph Construction**: Uses top 0.1% most similar transactions to build a graph
4. **Analysis**: Applies PageRank to identify hub transactions and communities

### Anomaly Detection Methods
- **Statistical Methods**: Z-score, IQR-based outlier detection
- **Time-based Analysis**: Unusual transaction timing patterns
- **Pattern Analysis**: Rapid successive transactions, geographic anomalies
- **Email Analysis**: Suspicious email patterns and domains

## 📈 Analysis Tools

### 1. **Comprehensive Analysis Script** (`analyze_transaction_data.py`)
- Complete data exploration and anomaly detection
- Generates visualizations and saves as PNG files
- Provides detailed recommendations

### 2. **Interactive Notebook** (`transaction_anomaly_analysis.ipynb`)
- Step-by-step analysis with explanations
- Interactive visualizations
- Perfect for exploring data patterns

### 3. **Original CTG Implementation** (`src/`)
- `gen_costumer_transaction_graph.py` - CTG generation
- `explore_ctg.py` - Graph visualization and analysis
- `pars_and_manipulate_data.py` - Data preprocessing
- `run_ctg_analysis.py` - Main execution script

## 🎨 Visualizations

The analysis generates comprehensive visualizations including:
- Transaction amount distributions
- Order status breakdowns
- Hourly transaction patterns
- IP and email frequency analysis
- Amount vs status relationships

## 🔮 Future Plans

### Phase 1: Enhanced Detection Methods
- [ ] **Isolation Forest** implementation for unsupervised anomaly detection
- [ ] **Local Outlier Factor (LOF)** for density-based anomalies
- [ ] **One-Class SVM** for complex pattern detection
- [ ] **Ensemble methods** combining multiple detection approaches

### Phase 2: Advanced Graph Analysis
- [ ] **Community detection algorithms** (Louvain, Leiden)
- [ ] **Graph neural networks** for deep learning-based detection
- [ ] **Dynamic graph analysis** for temporal patterns
- [ ] **Multi-layer graph construction** with different similarity metrics

### Phase 3: Feature Engineering
- [ ] **Temporal features**: Time since last transaction, day of week patterns
- [ ] **Geographic features**: IP geolocation, distance calculations
- [ ] **Behavioral features**: Transaction frequency, amount patterns
- [ ] **Email analysis**: Domain reputation, pattern recognition

### Phase 4: Production Pipeline
- [ ] **Real-time detection** system
- [ ] **API endpoints** for transaction scoring
- [ ] **Dashboard** for monitoring and alerts
- [ ] **Model versioning** and A/B testing framework

### Phase 5: Evaluation & Optimization
- [ ] **Labeled test datasets** with known anomalies
- [ ] **Performance metrics**: Precision, recall, F1-score
- [ ] **Cross-validation** and model selection
- [ ] **Hyperparameter optimization**

## 🛠️ Development

### Project Structure
```
FindingTransactionsHubs/
├── data/                           # Transaction datasets
├── src/                           # Original CTG implementation
├── test/                          # Test files
├── analyze_transaction_data.py    # Comprehensive analysis script
├── transaction_anomaly_analysis.ipynb  # Interactive notebook
├── pyproject.toml                 # Project configuration
├── uv.lock                       # Dependency lock file
└── README.md                     # This file
```

### Dependencies
- **Data Science**: pandas, numpy, matplotlib, seaborn
- **Machine Learning**: scikit-learn
- **Graph Analysis**: networkx
- **Notebooks**: jupyter, ipykernel
- **Development**: pytest, black, flake8, mypy

## 📝 Usage Examples

```bash
# Run full analysis with visualizations
uv run python analyze_transaction_data.py

# Start interactive analysis
uv run jupyter notebook transaction_anomaly_analysis.ipynb

# Run original CTG approach
uv run python src/run_ctg_analysis.py

# Run tests
uv run pytest test/
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

---

**Note**: This project is currently in development. The current CTG approach provides a solid foundation, but the planned enhancements will significantly improve detection accuracy and provide more sophisticated analysis capabilities.
