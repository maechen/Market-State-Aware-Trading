import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def baseline_strategy(
    ticker,
    start_date = "2007-01-01",
    end_date = "2020-06-11",
    initial_capital = 10000,
    strategy = ('b&h',)):
    """
    Parameters
    ----------
    ticker : str
        Ticker symbol of the asset (e.g., "SPY", "AAPL").
        
    start_date : str, optional
        Start date of the backtest in "YYYY-MM-DD" format.
        
    end_date : str, optional
        End date of the backtest in "YYYY-MM-DD" format.
        
    initial_capital : float, optional
        Initial portfolio value at the beginning of the backtest.
        
    strategy : tuple, optional
        Strategy specification.
        
        Supported formats:
        
        ('b&h',)
            Buy-and-hold strategy (always invested).
            
        ('momentum', lookback)
            Momentum strategy using lookback-period return.
            If past lookback return > 0 → invest (signal=1),
            else → stay in cash (signal=0).
            
            lookback must be a positive integer.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing:
        - price
        - daily_return
        - signal
        - strategy_return
        - cumulative
        - portfolio_value
        -* momentum (only for momentum strategy)
    """
    if not isinstance(strategy, tuple):
        raise TypeError("strategy must be a tuple")

    if len(strategy) == 0:
        raise ValueError("strategy tuple cannot be empty")

    if strategy[0] == "b&h":
        if len(strategy) != 1:
            raise ValueError("Buy & Hold strategy must be ('b&h',)")

    elif strategy[0] == "momentum":
        if len(strategy) != 2:
            raise ValueError("Momentum strategy must be ('momentum', lookback)")
        
        lookback = strategy[1]
        if not isinstance(lookback, int) or lookback <= 0:
            raise ValueError("Momentum lookback must be a positive integer")

    else:
        raise ValueError("Invalid strategy. Choose 'b&h' or 'momentum'.")

    data = yf.download(ticker, start=start_date, end=end_date)
    data = data['Close'].copy()
    data.rename(columns={'SPY':'price'}, inplace=True)
    data['daily_return'] = data['price'].pct_change()
    if strategy[0] == "b&h":
        data['signal'] = 1
    elif strategy[0] == "momentum":
        data['momentum'] = data['price'].pct_change(lookback)
        data['signal'] = np.where(data['momentum'] > 0, 1, 0)
    else:
        raise ValueError("Invalid strategy. Choose 'b&h' or 'momentum'.")

    data['signal'] = data['signal'].shift(1)
    data['strategy_return'] = data['signal'] * data['daily_return']
    data['cumulative'] = (1 + data['strategy_return']).cumprod()
    data['portfolio_value'] = initial_capital * data['cumulative']
    data.dropna(inplace=True)

    return data


if __name__ == "__main__":
    spy_bh = baseline_strategy('SPY', strategy=('b&h',))
    spy_mom = baseline_strategy('SPY', strategy=('momentum', 126))
    # Plot portfolio growth
    plt.figure()
    plt.plot(spy_bh.index, spy_bh['portfolio_value'], label='Buy & Hold')
    plt.plot(spy_mom.index, spy_mom['portfolio_value'], label='Momentum')
    plt.title("SPY Portfolio Growth")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.savefig('spy_portfolio_growth.png')