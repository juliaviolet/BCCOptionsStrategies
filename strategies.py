# strategies.py
strategies_content = """

import numpy as np
import pandas as pd

# Function to compute the adjusted shadow delta based on Taleb's approach
def compute_taleb_adjusted_shadow_delta(row, matching_row, include_theta=True):
    gamma_hedge = matching_row['GAMMA_HEDGE']
    delta_hedge = matching_row['DELTA_HEDGE']
    vega_hedge = matching_row['VEGA_HEDGE']
    theta_hedge = matching_row['THETA_HEDGE'] if include_theta else 0.0

    # Taleb's adjustment
    gamma_adjustment = gamma_hedge * (row['Price_Change']**2) / 2
    delta_hedge += gamma_adjustment

    return round(delta_hedge + gamma_hedge * row['Price_Change'] + vega_hedge * row['Volatility_Change'] + theta_hedge)

# Original function to compute the adjusted shadow delta taking into account the volatility smile
def compute_adjusted_shadow_delta(row, matching_row, include_theta=True):
    adjustment_factor = 0.05  # Factor to adjust the delta based on the volatility smile
    delta_hedge = matching_row['DELTA_HEDGE']
    gamma_hedge = matching_row['GAMMA_HEDGE']
    vega_hedge = matching_row['VEGA_HEDGE']
    theta_hedge = matching_row['THETA_HEDGE'] if include_theta else 0.0
    price_diff = row['CF_CLOSE_y'] - matching_row['STRIKE_PRC']

    # Adjust delta based on the volatility smile
    if matching_row['PUTCALLIND'] == 'CALL':
        if price_diff > 0:
            delta_hedge -= adjustment_factor * abs(price_diff)
        else:
            delta_hedge += adjustment_factor * abs(price_diff)
    else:
        if price_diff < 0:
            delta_hedge += adjustment_factor * abs(price_diff)
        else:
            delta_hedge -= adjustment_factor * abs(price_diff)

    return round(delta_hedge + gamma_hedge * row['Price_Change'] + vega_hedge * row['Volatility_Change'] + theta_hedge)

# Function to calculate hedging metrics based on different strategies
def calculate_hedging_metrics(final_results_data, strategies=['DELTA', 'VEGA', 'THETA', 'GAMMA'], shadow=False, adjust_for_smile=False, use_taleb_adjustment=False):
    final_results_data = final_results_data.iloc[1:].reset_index(drop=True)
    final_results_data['Volatility_Change'] = final_results_data['IMP_VOLT'].diff().fillna(0)

    initial_portfolio_value = 10000.00
    transaction_cost_per_contract = 0.75
    running_balance = initial_portfolio_value
    hedging_metrics_df = pd.DataFrame()

    for index, row in final_results_data.iterrows():
        metrics_row = row.to_dict()
        daily_profit = 0.0

        if shadow:
            matching_rows = final_results_data[final_results_data['Unnamed: 0'] == row['Unnamed: 0']]
            if not matching_rows.empty:
                matching_row = matching_rows.iloc[0]
                if use_taleb_adjustment:
                    delta_value = compute_taleb_adjusted_shadow_delta(row, matching_row, 'THETA' in strategies)
                elif adjust_for_smile:
                    delta_value = compute_adjusted_shadow_delta(row, matching_row, 'THETA' in strategies)
                else:
                    delta_value = row['DELTA_HEDGE'] + row['GAMMA_HEDGE'] * row['Price_Change'] + row['VEGA_HEDGE'] * row['Volatility_Change']
                    if 'THETA' in strategies:
                        delta_value += row['THETA_HEDGE']
                hedge_value = round(delta_value)
            else:
                hedge_value = sum(row[strategy + '_HEDGE_ROUND'] for strategy in strategies)
        else:
            hedge_value = sum(row[strategy + '_HEDGE_ROUND'] for strategy in strategies)

        transaction_value = -1 * hedge_value * row['CF_CLOSE_x']
        daily_buy_transaction_cost = abs(transaction_value) * transaction_cost_per_contract if transaction_value > 0 else 0
        daily_sell_transaction_cost = abs(transaction_value) * transaction_cost_per_contract if transaction_value < 0 else 0
        total_cost_revenue = transaction_value - daily_buy_transaction_cost - daily_sell_transaction_cost

        if total_cost_revenue > 0:
            daily_profit += total_cost_revenue

        running_balance += daily_profit
        metrics_row["Running_Balance"] = running_balance
        hedging_metrics_df = pd.concat([hedging_metrics_df, pd.DataFrame([metrics_row])], ignore_index=True)

    return hedging_metrics_df

# Ratio Spread Strategy: Aim to find the best combination of options to maximize profit
def find_ratio_spread_combination(options_data):
    best_combination = None
    best_profit = float('-inf')
    best_ratio = None
    options_list = options_data.to_dict(orient='records')
    for i, option_1 in enumerate(options_list):
        for j, option_2 in enumerate(options_list[i+1:], start=i+1):
            for ratio in range(1, 101):
                profit = ratio * option_1['CF_CLOSE_x'] - option_2['CF_CLOSE_x']
                if profit > best_profit:
                    best_profit = profit
                    best_combination = [option_1, option_2]
                    best_ratio = ratio
    return best_combination, best_ratio, best_profit

# Short Strangle Strategy: Involves selling out-of-the-money call and put options
def select_out_of_the_money_call(group):
    return group[group['STRIKE_PRC'] < group['CF_CLOSE_y']].nlargest(1, 'STRIKE_PRC')

def select_out_of_the_money_put(group):
    return group[group['STRIKE_PRC'] > group['CF_CLOSE_y']].nsmallest(1, 'STRIKE_PRC')
"""
with open('strategies.py', 'w') as file:
    file.write(strategies_content)
