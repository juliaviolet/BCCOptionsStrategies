utilities_content = """
import pandas as pd

def read_csv(file_name):
    return pd.read_csv(file_name)

def write_csv(df, file_name):
    df.to_csv(file_name, index=False)

def calculate_hedge_amounts(df):
    df['DELTA_HEDGE'] = df['DELTA'] * df['PRICE_CHANGE']
    df['VEGA_HEDGE'] = df['VEGA'] * df['PRICE_CHANGE']
    df['THETA_HEDGE'] = df['THETA'] * df['PRICE_CHANGE']
    df['GAMMA_HEDGE'] = df['GAMMA'] * df['PRICE_CHANGE']
"""

with open('utilities.py', 'w') as file:
    file.write(utilities_content)
