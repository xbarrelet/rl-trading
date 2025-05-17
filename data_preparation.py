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
        select open, high, low, close, volume, timestamp from quotes 
        where symbol = %s and interval = %s order by timestamp asc
        """, (symbol, timeperiod))
        return cursor.fetchall()


def load_data(symbol, timeperiod):
    quotes = load_quotes(symbol, timeperiod)

    df = DataFrame(quotes)

    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(float)

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

    add_indicators_to_df(df, timeperiod)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    print(f"Data shape after dropping NaN: {df.shape}")

    # Identify consecutive periods for better training
    df = identify_consecutive_periods(df)

    return df


def identify_consecutive_periods(df):
    """Identify consecutive periods in the data to handle gaps"""
    # Create a new column to mark the start of each consecutive period
    time_diff = df['timestamp'].diff()

    # Find standard time interval (most common difference)
    time_intervals = time_diff.value_counts()
    if not time_intervals.empty:
        std_interval = time_intervals.index[0]
        # Mark gaps that are more than 2x the standard interval
        df['period_start'] = (time_diff > std_interval * 2).astype(int)
        # Cumulative sum to mark each period
        df['period'] = df['period_start'].cumsum()
        # Drop the temporary column - only if it exists
        if 'period_start' in df.columns:
            df.drop('period_start', axis=1, inplace=True)
    else:
        # If we can't determine the standard interval, just use a single period
        df['period'] = 0

    return df


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
    volume = df['volume'].values

    if timeperiod == 30:
        multiplier = 2
    elif timeperiod == 15:
        multiplier = 4
    elif timeperiod == 5:
        multiplier = 12
    elif timeperiod == 1:
        multiplier = 60
    else:
        multiplier = 1

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

    # ===== 1. MARKET REGIME IDENTIFICATION =====

    # Directional Movement Index (DMI/ADX) - Trend strength indicator
    df['ADX'] = talib.ADX(high=df['high'].values if 'high' in df.columns else close,
                          low=df['low'].values if 'low' in df.columns else close * 0.998,
                          close=close,
                          timeperiod=14 * multiplier)

    # ADX interpretation helper columns
    df['is_strong_trend'] = df['ADX'] > 25  # Boolean: True if strong trend

    # Aroon indicator - identifies trend changes
    df['aroon_up'], df['aroon_down'] = talib.AROON(
        high=df['high'].values if 'high' in df.columns else close * 1.002,
        low=df['low'].values if 'low' in df.columns else close * 0.998,
        timeperiod=14 * multiplier
    )
    df['aroon_oscillator'] = df['aroon_up'] - df['aroon_down']

    # Hilbert Transform - Dominant Cycle Period
    df['HT_DCPERIOD'] = talib.HT_DCPERIOD(close)

    # Detect market regimes using SMA cross
    df['SMA_fast'] = talib.SMA(close, timeperiod=10 * multiplier)
    df['SMA_slow'] = talib.SMA(close, timeperiod=40 * multiplier)
    df['SMA_diff'] = df['SMA_fast'] - df['SMA_slow']
    df['SMA_diff_pct'] = df['SMA_diff'] / df['SMA_slow'] * 100

    # Market regime: 1 = uptrend, 0 = sideways/unclear, -1 = downtrend
    # More conservative market regime classification with wider thresholds
    # Modify this section in data_preparation.py
    df['market_regime'] = 0
    df.loc[df['SMA_diff_pct'] > 0.5, 'market_regime'] = 1  # More lenient uptrend condition
    df.loc[df['SMA_diff_pct'] < -0.5, 'market_regime'] = -1  # More lenient downtrend condition

    # For stronger classifications, add this secondary condition
    df.loc[(df['SMA_diff_pct'] > 1.0) & (df['ADX'] > 25), 'market_regime'] = 2  # Strong uptrend
    df.loc[(df['SMA_diff_pct'] < -1.0) & (df['ADX'] > 25), 'market_regime'] = -2  # Strong downtrend

    # Chande Momentum Oscillator - detects overbought/oversold
    df['CMO'] = talib.CMO(close, timeperiod=14 * multiplier)

    # ===== 2. VOLATILITY MEASURES =====

    # Average True Range (ATR) - volatility indicator
    if 'high' in df.columns and 'low' in df.columns:
        df['ATR'] = talib.ATR(
            high=df['high'].values,
            low=df['low'].values,
            close=close,
            timeperiod=14 * multiplier
        )

        # Normalized ATR (ATR as percentage of price)
        df['ATR_pct'] = df['ATR'] / close * 100
    else:
        # Approximate ATR using close prices
        high_approx = close * 1.002  # Approximate high as 0.2% above close
        low_approx = close * 0.998  # Approximate low as 0.2% below close
        df['ATR'] = talib.ATR(
            high=high_approx,
            low=low_approx,
            close=close,
            timeperiod=14 * multiplier
        )
        df['ATR_pct'] = df['ATR'] / close * 100

    # Historical volatility using standard deviation
    df['volatility_5'] = df['close'].pct_change().rolling(window=5 * multiplier).std() * 100
    df['volatility_20'] = df['close'].pct_change().rolling(window=20 * multiplier).std() * 100

    # Bollinger Band width - another volatility measure
    _, _, _ = talib.BBANDS(close, timeperiod=20 * multiplier, nbdevup=2, nbdevdn=2, matype=0)
    df['bb_width'] = (df['upper_bb'] - df['lower_bb']) / df['middle_bb'] * 100

    # Rate of change of volatility
    df['volatility_change'] = df['volatility_20'].pct_change(periods=5 * multiplier) * 100

    # ===== 3. VOLUME INDICATORS =====

    if 'volume' in df.columns:
        # On-Balance Volume (OBV)
        df['OBV'] = talib.OBV(close, volume)

        # Normalized OBV - makes it more comparable across time
        df['OBV_norm'] = df['OBV'] / df['OBV'].rolling(window=20 * multiplier).mean()

        # Money Flow Index (MFI) - volume-weighted RSI
        if 'high' in df.columns and 'low' in df.columns:
            df['MFI'] = talib.MFI(
                high=df['high'].values,
                low=df['low'].values,
                close=close,
                volume=volume,
                timeperiod=14 * multiplier
            )
        else:
            df['MFI'] = talib.MFI(
                high=high_approx,
                low=low_approx,
                close=close,
                volume=volume,
                timeperiod=14 * multiplier
            )

        # Chaikin A/D Oscillator
        if 'high' in df.columns and 'low' in df.columns:
            df['ADOSC'] = talib.ADOSC(
                high=df['high'].values,
                low=df['low'].values,
                close=close,
                volume=volume,
                fastperiod=3 * multiplier,
                slowperiod=10 * multiplier
            )
        else:
            df['ADOSC'] = talib.ADOSC(
                high=high_approx,
                low=low_approx,
                close=close,
                volume=volume,
                fastperiod=3 * multiplier,
                slowperiod=10 * multiplier
            )

        # Volume Rate of Change
        df['volume_roc'] = talib.ROC(volume, timeperiod=1)

        # Volume vs Moving Average Volume
        df['volume_sma'] = talib.SMA(volume, timeperiod=20 * multiplier)
        df['volume_ratio'] = volume / df['volume_sma']

        # Price-Volume Trend
        df['PVT'] = (df['close'].pct_change() * volume).cumsum()
