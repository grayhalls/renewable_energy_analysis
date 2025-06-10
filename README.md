# Sustainable Energy Analysis Dashboard

This Streamlit application provides an interactive dashboard for analyzing global sustainable energy data. The app visualizes trends in renewable energy adoption, electricity access, and energy mix across different regions.

## Features

- Interactive year range filter
- Renewable energy share trends
- Global electricity access visualization
- Regional energy mix analysis
- Data summary statistics

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Set up Kaggle API credentials:
   - Go to your Kaggle account settings
   - Create a new API token
   - Place the `kaggle.json` file in `~/.kaggle/` (Linux/Mac) or `C:\Users\<Windows-username>\.kaggle\` (Windows)

3. Run the Streamlit app:
```bash
streamlit run app.py
```

## Data Source

The data is sourced from the "Global Data on Sustainable Energy" dataset on Kaggle, which provides comprehensive information about sustainable energy metrics across different countries and regions.

## Requirements

- Python 3.8+
- Streamlit
- Pandas
- Plotly
- Kagglehub
- NumPy 
- Statsmodels