import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Load datasets from CSV files
final_results_data = pd.read_csv('final_results_price_rounded_2.csv')
combined_hedging_metrics = pd.read_csv('combined_hedging_metrics_rounded_2.csv')
sorted_updated_merged_data = pd.read_csv('sorted_updated_merged_data_fixed.csv')


def compute_taleb_adjusted_shadow_delta(row, matching_row, include_theta=True):
    """
    Compute the adjusted shadow delta based on "Taleb's approach".
    """
    gamma_hedge = matching_row['GAMMA_HEDGE']
    delta_hedge = matching_row['DELTA_HEDGE']
    vega_hedge = matching_row['VEGA_HEDGE']
    theta_hedge = matching_row['THETA_HEDGE'] if include_theta else 0.0

    # Taleb's adjustment
    gamma_adjustment = gamma_hedge * (row['Price_Change'] ** 2) / 2
    delta_hedge += gamma_adjustment

    return round(
        delta_hedge + gamma_hedge * row['Price_Change'] +
        vega_hedge * row['Volatility_Change'] + theta_hedge
    )


def compute_adjusted_shadow_delta(row, matching_row, include_theta=True):
    """
    Compute the adjusted shadow delta taking into account the volatility smile.
    """
    adjustment_factor = 0.05  # factor to adjust the delta
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

    return round(
        delta_hedge + gamma_hedge * row['Price_Change'] +
        vega_hedge * row['Volatility_Change'] + theta_hedge
    )


def calculate_hedging_metrics(final_results_data,
                              strategies=['DELTA', 'VEGA', 'THETA', 'GAMMA'],
                              shadow=False, adjust_for_smile=False,
                              use_taleb_adjustment=False):
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
                    delta_value = (
                        row['DELTA_HEDGE'] +
                        row['GAMMA_HEDGE'] * row['Price_Change'] +
                        row['VEGA_HEDGE'] * row['Volatility_Change']
                    )
                    if 'THETA' in strategies:
                        delta_value += row['THETA_HEDGE']
                hedge_value = round(delta_value)
            else:
                hedge_value = sum(row[strategy + '_HEDGE_ROUND'] for strategy in strategies)
        else:
            hedge_value = sum(row[strategy + '_HEDGE_ROUND'] for strategy in strategies)

        transaction_value = -1 * hedge_value * row['CF_CLOSE_x']
        daily_buy_transaction_cost = (
            abs(transaction_value) * transaction_cost_per_contract
            if transaction_value > 0 else 0
        )
        daily_sell_transaction_cost = (
            abs(transaction_value) * transaction_cost_per_contract
            if transaction_value < 0 else 0
        )
        total_cost_revenue = transaction_value - daily_buy_transaction_cost - daily_sell_transaction_cost

        if total_cost_revenue > 0:
            daily_profit += total_cost_revenue

        running_balance += daily_profit
        metrics_row["Running_Balance"] = running_balance
        hedging_metrics_df = pd.concat([hedging_metrics_df, pd.DataFrame([metrics_row])], ignore_index=True)

    return hedging_metrics_df

# compute metrics for multiple strategies including delta, vega, theta,
# gamma, and shadow delta
delta_theta_metrics = calculate_hedging_metrics(
    final_results_data, strategies=['DELTA', 'VEGA', 'THETA', 'GAMMA']
)
delta_metrics = calculate_hedging_metrics(
    final_results_data, strategies=['DELTA', 'VEGA', 'GAMMA']
)
shadow_delta_metrics = calculate_hedging_metrics(
    final_results_data, strategies=['DELTA', 'VEGA', 'THETA', 'GAMMA'], shadow=True
)
shadow_delta_without_theta_metrics = calculate_hedging_metrics(
    final_results_data, strategies=['DELTA', 'VEGA', 'GAMMA'], shadow=True
)
adjusted_shadow_delta_metrics = calculate_hedging_metrics(
    final_results_data,
    strategies=['DELTA', 'VEGA', 'THETA', 'GAMMA'],
    shadow=True,
    adjust_for_smile=True
)
adjusted_shadow_delta_without_theta_metrics = calculate_hedging_metrics(
    final_results_data,
    strategies=['DELTA', 'VEGA', 'GAMMA'],
    shadow=True,
    adjust_for_smile=True
)


# Ratio Spread Strategy: Aim to find the best combination of options
# to maximize profit
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

# filter and sort the data to find the best combinations for calls and puts
filtered_data = sorted_updated_merged_data[
    (sorted_updated_merged_data['EXPIR_DATE'] == '2020-06-19') &
    (sorted_updated_merged_data['IMP_VOLT'] > 0)
]

# separate the data into calls and puts and sort by strike price
call_data_ratio = filtered_data[filtered_data['PUTCALLIND'] == 'CALL'].sort_values(by='STRIKE_PRC')
put_data_ratio = filtered_data[filtered_data['PUTCALLIND'] == 'PUT '].sort_values(by='STRIKE_PRC')
best_call_combination, best_call_ratio, best_call_profit = find_ratio_spread_combination(call_data_ratio)
best_put_combination, best_put_ratio, best_put_profit = find_ratio_spread_combination(put_data_ratio)

# married put strategy: protective strategy involving buying a put option for a stock owned
put_options = combined_hedging_metrics[combined_hedging_metrics['PUTCALLIND'] == 'PUT '].copy()
put_options['stock_cost'] = put_options['CF_CLOSE_y'] * 100
put_options['option_cost'] = put_options['CF_CLOSE_x'] * 1
put_options['total_cost'] = put_options['stock_cost'] + put_options['option_cost']
put_options['stock_profit_next_day'] = (put_options['CF_CLOSE_y'].shift(-1) * 100) - put_options['stock_cost']
put_options['stock_price_next_day'] = put_options['CF_CLOSE_y'].shift(-1)
put_options['option_profit'] = put_options.apply(
    lambda row: max(0, row['STRIKE_PRC'] - row['stock_price_next_day']) * 100 - row['option_cost'], axis=1
)
put_options['net_profit'] = put_options['stock_profit_next_day'] + put_options['option_profit']


# short strangle strategy: involves selling out-of-the-money call and put options
def select_out_of_the_money_call(group):
    return group[group['STRIKE_PRC'] < group['CF_CLOSE_y']].nlargest(1, 'STRIKE_PRC')


def select_out_of_the_money_put(group):
    return group[group['STRIKE_PRC'] > group['CF_CLOSE_y']].nsmallest(1, 'STRIKE_PRC')


# filter and sort the data, then calculate profits based on the strategy
df_clean = combined_hedging_metrics.dropna(subset=['CF_CLOSE_x', 'CF_CLOSE_y'])
calls_df_clean = df_clean[df_clean['PUTCALLIND'] == 'CALL']
puts_df_clean = df_clean[df_clean['PUTCALLIND'] == 'PUT ']

out_of_the_money_calls = (calls_df_clean.groupby('CF_DATE')
                          .apply(select_out_of_the_money_call)
                          .dropna().reset_index(drop=True))

out_of_the_money_puts = (puts_df_clean.groupby('CF_DATE')
                         .apply(select_out_of_the_money_put)
                         .dropna().reset_index(drop=True))

long_strangle_df = pd.merge(out_of_the_money_calls, out_of_the_money_puts,
                            on='CF_DATE', suffixes=('_call', '_put'))

df_sorted = df_clean.sort_values(by="CF_DATE")
df_sorted['CF_CLOSE_y_next'] = df_sorted['CF_CLOSE_y'].shift(-1)
long_strangle_with_next_price = pd.merge(long_strangle_df,
                                         df_sorted[['CF_DATE', 'CF_CLOSE_y_next']],
                                         on='CF_DATE', how='left')

long_strangle_with_next_price['profit_from_call_sold'] = np.where(
    long_strangle_with_next_price['CF_CLOSE_y_next'] > long_strangle_with_next_price['STRIKE_PRC_call'],
    -((long_strangle_with_next_price['CF_CLOSE_y_next'] -
       long_strangle_with_next_price['STRIKE_PRC_call']) * 100) +
    (long_strangle_with_next_price['CF_CLOSE_x_call'] * 100),
    long_strangle_with_next_price['CF_CLOSE_x_call'] * 100)

long_strangle_with_next_price['profit_from_put_sold'] = np.where(
    long_strangle_with_next_price['CF_CLOSE_y_next'] < long_strangle_with_next_price['STRIKE_PRC_put'],
    -((long_strangle_with_next_price['STRIKE_PRC_put'] -
       long_strangle_with_next_price['CF_CLOSE_y_next']) * 100) +
    (long_strangle_with_next_price['CF_CLOSE_x_put'] * 100),
    long_strangle_with_next_price['CF_CLOSE_x_put'] * 100)

long_strangle_with_next_price['total_profit_loss_sold'] = (long_strangle_with_next_price['profit_from_call_sold'] +
                                                           long_strangle_with_next_price['profit_from_put_sold'])

taleb_shadow_delta_metrics = calculate_hedging_metrics(
    final_results_data, strategies=['DELTA', 'VEGA', 'THETA', 'GAMMA'],
    shadow=True, use_taleb_adjustment=True)

taleb_shadow_delta_without_theta_metrics = calculate_hedging_metrics(
    final_results_data, strategies=['DELTA', 'VEGA', 'GAMMA'],
    shadow=True, use_taleb_adjustment=True)

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

# sort the strategies based on their final balance
sorted_balances = {k: v for k, v in sorted(balances.items(),
                                           key=lambda item: item[1],
                                           reverse=True)}

# plot the results using Plotly
strategies = list(sorted_balances.keys())
values = list(sorted_balances.values())

fig = go.Figure(data=[go.Bar(
    x=strategies,
    y=values,
    marker=dict(
        color=values,
        colorscale='viridis',
        colorbar=dict(title='Balance')
    )
)])
fig.update_layout(
    title="Balances Across Different Hedging Strategies",
    xaxis_title="Strategy",
    yaxis_title="Balance",
    template="plotly",
    xaxis_tickangle=-45,
    yaxis=dict(tickformat="$,.2f")
)
fig.show()

# print the final balance for each strategy
for strategy, balance in sorted_balances.items():
    print(f"{strategy}: ${balance:,.2f}")
