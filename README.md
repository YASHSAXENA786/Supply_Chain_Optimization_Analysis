# Supply Chain Optimization Analysis

This project provides a complete supply chain analytics solution, including data generation, exploratory data analysis (EDA), demand forecasting, and a modern Streamlit dashboard for interactive visualization.

## Features
- **Synthetic Data Generation**: Generate realistic supply chain datasets (products, suppliers, sales, inventory, purchase orders).
- **Exploratory Data Analysis (EDA)**: Analyze inventory performance, supplier reliability, cost optimization, and more.
- **Demand Forecasting**: Machine learning-based demand forecasting for top products.
- **Unified Streamlit Dashboard**: Visualize and interact with all data and analytics in one place.

## How to Use
1. **Clone the repository**
	```bash
	git clone https://github.com/YASHSAXENA786/Supply_Chain_Optimization_Analysis.git
	cd Supply_Chain_Optimization_Analysis
	```
2. **Set up the Python environment**
	```bash
	python -m venv .venv
	.venv\Scripts\activate  # On Windows
	pip install -r requirements.txt  # Or install: pandas, numpy, matplotlib, seaborn, scikit-learn, streamlit
	```
3. **Generate Data**
	```bash
	python supply_chain_data_gen.py
	```
4. **Run EDA or Forecasting (optional)**
	```bash
	python supply_chain_eda.py
	python demand_forecasting.py
	```
5. **Launch the Dashboard**
	```bash
	streamlit run supply_chain_dashboard.py
	```
	Open the local URL provided by Streamlit in your browser.

## Project Structure
- `supply_chain_data_gen.py` — Data generation script
- `supply_chain_eda.py` — Exploratory data analysis
- `demand_forecasting.py` — Demand forecasting with ML
- `supply_chain_dashboard.py` — Streamlit dashboard
- `*.csv` — Generated datasets and analysis results

## Requirements
- Python 3.8+
- pandas, numpy, matplotlib, seaborn, scikit-learn, streamlit

## License
MIT License

---
*Created by Yash Saxena*
# Supply_Chain_Optimization_Analysis