import os
import shutil

import numpy as np
import pandas as pd
import psycopg
import talib
from pandas import DataFrame
from psycopg.rows import dict_row

connection_url = 'postgresql://postgres:postgres123@localhost:5429/quotes'
db_connection = psycopg.connect(connection_url, row_factory=dict_row, autocommit=True)

pd.options.mode.copy_on_write = True


def load_quotes(symbol, timeperiod):
    with db_connection.cursor() as cursor:
        cursor.execute("""
        select close, timestamp from quotes 
        where symbol = %s and interval = %s order by timestamp asc
        """, (symbol, timeperiod))
        return cursor.fetchall()


def load_quotes(symbol, timeperiod):
    with db_connection.cursor() as cursor:
        cursor.execute("""
        select close, timestamp from quotes 
        where symbol = %s and interval = %s order by timestamp asc
        """, (symbol, timeperiod))
        return cursor.fetchall()


def load_data(symbol, timeperiod):
    quotes = load_quotes(symbol, timeperiod)

    df = DataFrame(quotes)
    df['close'] = df['close'].astype(float)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

    return df


def identify_consecutive_periods(df, time_col, freq):
    # Create a complete date range based on the min and max timestamps
    complete_range = pd.date_range(start=df[time_col].min(), end=df[time_col].max(), freq=freq)

    # Find the missing timestamps
    missing_timestamps = complete_range.difference(df[time_col])

    # Create a new column to mark the start of each consecutive period
    df['period'] = (df[time_col].diff() > pd.Timedelta(freq)).cumsum()

    return df, missing_timestamps


def get_period_start_end(df, time_col):
    period_ranges = df.groupby('period')[time_col].agg(['min', 'max']).reset_index()
    period_ranges.columns = ['period', 'start_timestamp', 'end_timestamp']
    return period_ranges


def get_latest_period(df):
    latest_period = df['period'].max()
    latest_period_df = df[df['period'] == latest_period]
    latest_period_df.drop(columns=['period'], inplace=True)

    return latest_period_df


def add_indicators_to_df(df, timeperiod=60):
    close = df['close'].values

    if timeperiod == 30:
        multiplier = 2
    elif timeperiod == 15:
        multiplier = 4
    elif timeperiod == 5:
        multiplier = 12
    elif timeperiod == 1:
        multiplier = 60

    # Trend Indicators
    df['SMA_10'] = talib.SMA(close, timeperiod=10 * multiplier)
    df['EMA_10'] = talib.EMA(close, timeperiod=10 * multiplier)
    df['SMA_20'] = talib.SMA(close, timeperiod=20 * multiplier)
    df['EMA_20'] = talib.EMA(close, timeperiod=20 * multiplier)
    df['SMA_50'] = talib.SMA(close, timeperiod=50 * multiplier)
    df['EMA_50'] = talib.EMA(close, timeperiod=50 * multiplier)
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(close,
                                                                fastperiod=12 * multiplier,
                                                                slowperiod=26 * multiplier,
                                                                signalperiod=9 * multiplier)
    # Momentum Indicators
    df['RSI'] = talib.RSI(close, timeperiod=14 * multiplier)
    df['ROC'] = talib.ROC(close, timeperiod=10 * multiplier)
    df['MOM'] = talib.MOM(close, timeperiod=10 * multiplier)

    # Volatility Indicators
    df['upper_bb'], df['middle_bb'], df['lower_bb'] = talib.BBANDS(close,
                                                                   timeperiod=20 * multiplier,
                                                                   nbdevup=2,
                                                                   nbdevdn=2,
                                                                   matype=0)

    # Additional indicators
    df['TRIX'] = talib.TRIX(close, timeperiod=30 * multiplier)
    df['HT_TRENDLINE'] = talib.HT_TRENDLINE(close)

    # Add price change percentage
    df['price_change_pct'] = df['close'].pct_change()
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))

    # # Add rolling mean of some indicators
    df['RSI_MA_14'] = df['RSI'].rolling(window=14).mean()
    df['MACD_MA_14'] = df['MACD'].rolling(window=14).mean()
