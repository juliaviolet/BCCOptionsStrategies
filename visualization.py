# visualization.py
visualization_content = """
import plotly.graph_objects as go

def plot_strategy_balances(sorted_balances):
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

"""
with open('visualization.py', 'w') as file:
    file.write(visualization_content)
