# main_2.py
main_2_content = """
from data_loader import *
from strategies import *
from visualization import plot_strategy_balances

def compute_strategies(final_results_data, combined_hedging_metrics, sorted_updated_merged_data):
    # Filter and sort the data, then calculate profits based on the strategy
    df_clean = combined_hedging_metrics.dropna(subset=['CF_CLOSE_x', 'CF_CLOSE_y'])
    calls_df_clean = df_clean[df_clean['PUTCALLIND'] == 'CALL']
    puts_df_clean = df_clean[df_clean['PUTCALLIND'] == 'PUT ']
    out_of_the_money_calls = calls_df_clean.groupby('CF_DATE').apply(select_out_of_the_money_call).dropna().reset_index(drop=True)
    out_of_the_money_puts = puts_df_clean.groupby('CF_DATE').apply(select_out_of_the_money_put).dropna().reset_index(drop=True)
    long_strangle_df = pd.merge(out_of_the_money_calls, out_of_the_money_puts, on='CF_DATE', suffixes=('_call', '_put'))
    df_sorted = df_clean.sort_values(by="CF_DATE")
    df_sorted['CF_CLOSE_y_next'] = df_sorted['CF_CLOSE_y'].shift(-1)
    long_strangle_with_next_price = pd.merge(long_strangle_df, df_sorted[['CF_DATE', 'CF_CLOSE_y_next']], on='CF_DATE', how='left')
    long_strangle_with_next_price['profit_from_call_sold'] = np.where(
        long_strangle_with_next_price['CF_CLOSE_y_next'] > long_strangle_with_next_price['STRIKE_PRC_call'],
        -((long_strangle_with_next_price['CF_CLOSE_y_next'] - long_strangle_with_next_price['STRIKE_PRC_call']) * 100) + (long_strangle_with_next_price['CF_CLOSE_x_call'] * 100),
        long_strangle_with_next_price['CF_CLOSE_x_call'] * 100)
    long_strangle_with_next_price['profit_from_put_sold'] = np.where(
        long_strangle_with_next_price['CF_CLOSE_y_next'] < long_strangle_with_next_price['STRIKE_PRC_put'],
        -((long_strangle_with_next_price['STRIKE_PRC_put'] - long_strangle_with_next_price['CF_CLOSE_y_next']) * 100) + (long_strangle_with_next_price['CF_CLOSE_x_put'] * 100),
        long_strangle_with_next_price['CF_CLOSE_x_put'] * 100)
    long_strangle_with_next_price['total_profit_loss_sold'] = long_strangle_with_next_price['profit_from_call_sold'] + long_strangle_with_next_price['profit_from_put_sold']
    taleb_shadow_delta_metrics = calculate_hedging_metrics(final_results_data, strategies=['DELTA', 'VEGA', 'THETA', 'GAMMA'], shadow=True, use_taleb_adjustment=True)
    taleb_shadow_delta_without_theta_metrics = calculate_hedging_metrics(final_results_data, strategies=['DELTA', 'VEGA', 'GAMMA'], shadow=True, use_taleb_adjustment=True)

    # Compute metrics for multiple strategies including delta, vega, theta, gamma, and shadow delta
    delta_theta_metrics = calculate_hedging_metrics(final_results_data, strategies=['DELTA', 'VEGA', 'THETA', 'GAMMA'])
    delta_metrics = calculate_hedging_metrics(final_results_data, strategies=['DELTA', 'VEGA', 'GAMMA'])
    shadow_delta_metrics = calculate_hedging_metrics(final_results_data, strategies=['DELTA', 'VEGA', 'THETA', 'GAMMA'], shadow=True)
    shadow_delta_without_theta_metrics = calculate_hedging_metrics(final_results_data, strategies=['DELTA', 'VEGA', 'GAMMA'], shadow=True)
    adjusted_shadow_delta_metrics = calculate_hedging_metrics(final_results_data, strategies=['DELTA', 'VEGA', 'THETA', 'GAMMA'], shadow=True, adjust_for_smile=True)
    adjusted_shadow_delta_without_theta_metrics = calculate_hedging_metrics(final_results_data, strategies=['DELTA', 'VEGA', 'GAMMA'], shadow=True, adjust_for_smile=True)

    # Filter and sort the data to find the best combinations for calls and puts
    filtered_data = sorted_updated_merged_data[(sorted_updated_merged_data['EXPIR_DATE'] == '2020-06-19') & (sorted_updated_merged_data['IMP_VOLT'] > 0)]

    # Separate the data into calls and puts and sort by strike price
    call_data_ratio = filtered_data[filtered_data['PUTCALLIND'] == 'CALL'].sort_values(by='STRIKE_PRC')
    put_data_ratio = filtered_data[filtered_data['PUTCALLIND'] == 'PUT '].sort_values(by='STRIKE_PRC')
    best_call_combination, best_call_ratio, best_call_profit = find_ratio_spread_combination(call_data_ratio)
    best_put_combination, best_put_ratio, best_put_profit = find_ratio_spread_combination(put_data_ratio)

    # Married Put Strategy: Protective strategy involving buying a put option for a stock owned
    put_options = combined_hedging_metrics[combined_hedging_metrics['PUTCALLIND'] == 'PUT '].copy()
    put_options['stock_cost'] = put_options['CF_CLOSE_y'] * 100
    put_options['option_cost'] = put_options['CF_CLOSE_x'] * 1
    put_options['total_cost'] = put_options['stock_cost'] + put_options['option_cost']
    put_options['stock_profit_next_day'] = (put_options['CF_CLOSE_y'].shift(-1) * 100) - put_options['stock_cost']
    put_options['stock_price_next_day'] = put_options['CF_CLOSE_y'].shift(-1)
    put_options['option_profit'] = put_options.apply(lambda row: max(0, row['STRIKE_PRC'] - row['stock_price_next_day']) * 100 - row['option_cost'], axis=1)
    put_options['net_profit'] = put_options['stock_profit_next_day'] + put_options['option_profit']

    balances = {
        "Delta Theta": delta_theta_metrics['Running_Balance'].iloc[-1],
        "Delta": delta_metrics['Running_Balance'].iloc[-1],
        "Shadow Delta": shadow_delta_metrics['Running_Balance'].iloc[-1],
        "Shadow Delta Without Theta": shadow_delta_without_theta_metrics['Running_Balance'].iloc[-1],
        "Adjusted Shadow Delta": adjusted_shadow_delta_metrics['Running_Balance'].iloc[-1],
        "Adjusted Shadow Delta Without Theta": adjusted_shadow_delta_without_theta_metrics['Running_Balance'].iloc[-1],
        "Taleb Shadow Delta": taleb_shadow_delta_metrics['Running_Balance'].iloc[-1],
        "Taleb Shadow Delta Without Theta": taleb_shadow_delta_without_theta_metrics['Running_Balance'].iloc[-1],
        "Ratio Spread (Calls)": best_call_profit,
        "Ratio Spread (Puts)": best_put_profit,
        "Married Put": put_options['net_profit'].sum(),
        "Short Strangle": long_strangle_with_next_price['total_profit_loss_sold'].sum()
    }

    # Sort the strategies based on their final balance
    sorted_balances = {k: v for k, v in sorted(balances.items(), key=lambda item: item[1], reverse=True)}
    return sorted_balances

if __name__ == "__main__":
    final_results_data = load_final_results()
    combined_hedging_metrics = load_combined_hedging_metrics()
    sorted_updated_merged_data = load_sorted_updated_merged_data()

    sorted_balances = compute_strategies(final_results_data, combined_hedging_metrics, sorted_updated_merged_data)

    plot_strategy_balances(sorted_balances)

"""
with open('main_2.py', 'w') as file:
    file.write(main_2_content)

# Return paths for the files saved
saved_files = {
    "data_loader.py": "data_loader.py",
    "strategies.py": "strategies.py",
    "visualization.py": "visualization.py",
    "main_2.py": "main_2.py"
}

saved_files
