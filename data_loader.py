# data_loader.py
data_loader_content = """
import pandas as pd

def load_final_results():
    return pd.read_csv('final_results_fixed_price_rounded_check_vega.csv')

def load_combined_hedging_metrics():
    return pd.read_csv('combined_hedging_metrics_rounded_check_vega.csv')

def load_sorted_updated_merged_data():
    return pd.read_csv('sorted_updated_merged_data_fixed.csv')
"""
with open('data_loader.py', 'w') as file:
    file.write(data_loader_content)
