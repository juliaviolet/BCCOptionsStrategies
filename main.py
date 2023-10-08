main_content = """
from model import *
from utilities import *

def compute_and_save():

  # Read the data
  merged_data = read_csv('sorted_updated_merged_data_fixed.csv')

  # Compute model prices and Greek values using the BCC model
  # Use the model functions on the merged_data dataframe as shown in the provided code

  # Calculate the price change and its absolute value for the underlying asset
  merged_data['PRICE_CHANGE'] = merged_data['CF_CLOSE_y'].diff()
  merged_data['PRICE_CHANGE_ABS'] = merged_data['PRICE_CHANGE'].abs()

  # Compute model prices and Greek values using the BCC model
  merged_data['MODEL_PRICE'] = merged_data.apply(lambda row: BCC_call_value_nojit(row['CF_CLOSE_y'], row['STRIKE_PRC'], row['T'], -0.00067, 3.76807610e+00, 1.00000000e-03, 8.77986615e-03, 5.03063522e-01, 0.00000000e+00, 3.95979221e-01, 7.10854746e-02, 4.01539560e-02, 3.76807610e+00, 1.00000000e-03, 8.77986615e-03, 6.99623759e-02, 9.99763692e-02, 5.00002718e-01), axis=1)
  merged_data['DELTA'] = merged_data.apply(lambda row: BCC_delta_nojit(row['CF_CLOSE_y'], row['STRIKE_PRC'], row['T'], r0, 3.76807610e+00, 1.00000000e-03, 8.77986615e-03, 5.03063522e-01, 0.00000000e+00, 3.95979221e-01, 7.10854746e-02, 4.01539560e-02, 3.76807610e+00, 1.00000000e-03, 8.77986615e-03, 6.99623759e-02, 9.99763692e-02, 5.00002718e-01, row['PRICE_CHANGE_ABS'], row['PUTCALLIND']), axis=1)
  merged_data['VEGA'] = merged_data.apply(lambda row: BCC_vega_nojit(row['CF_CLOSE_y'], row['STRIKE_PRC'], row['T'], -0.00067, 3.76807610e+00, 1.00000000e-03, 8.77986615e-03, 5.03063522e-01, 0.00000000e+00, 3.95979221e-01, 7.10854746e-02, 4.01539560e-02, 3.76807610e+00, 1.00000000e-03, 8.77986615e-03, 6.99623759e-02, 9.99763692e-02, 5.00002718e-01, row['PRICE_CHANGE_ABS']), axis=1)
  merged_data['THETA'] = merged_data.apply(lambda row: BCC_theta_nojit(row['CF_CLOSE_y'], row['STRIKE_PRC'], row['T'], -0.00067, 3.76807610e+00, 1.00000000e-03, 8.77986615e-03, 5.03063522e-01, 0.00000000e+00, 3.95979221e-01, 7.10854746e-02, 4.01539560e-02, 3.76807610e+00, 1.00000000e-03, 8.77986615e-03, 6.99623759e-02, 9.99763692e-02, 5.00002718e-01), axis=1)
  merged_data['GAMMA'] = merged_data.apply(lambda row: BCC_gamma_nojit(row['CF_CLOSE_y'], row['STRIKE_PRC'], row['T'], -0.00067, 3.76807610e+00, 1.00000000e-03, 8.77986615e-03, 5.03063522e-01, 0.00000000e+00, 3.95979221e-01, 7.10854746e-02, 4.01539560e-02, 3.76807610e+00, 1.00000000e-03, 8.77986615e-03, 6.99623759e-02, 9.99763692e-02, 5.00002718e-01, row['PRICE_CHANGE_ABS'], row['PUTCALLIND']), axis=1)

  # Calculate hedge amounts
  calculate_hedge_amounts(merged_data)

  # Extract relevant columns for final results
  final_columns = ['Unnamed: 0', 'Instrument', 'CF_DATE', 'EXPIR_DATE', 'PUTCALLIND', 'STRIKE_PRC', 'CF_CLOSE_x', 'IMP_VOLT', 'CF_CLOSE_y',
                  'DELTA', 'VEGA', 'THETA', 'GAMMA', 'DELTA_HEDGE', 'VEGA_HEDGE', 'THETA_HEDGE', 'GAMMA_HEDGE']
  final_df = merged_data[final_columns]

  # Save the final results
  write_csv(final_df, 'final_results_fixed.csv')
  return merged_data
  """

with open('main.py', 'w') as file:
    file.write(main_content)
