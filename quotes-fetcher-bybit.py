from datetime import datetime

import psycopg
from psycopg.rows import dict_row
from pybit.unified_trading import HTTP

connection_url = 'postgresql://postgres:postgres123@localhost:5429/quotes'
db_connection = psycopg.connect(connection_url, row_factory=dict_row, autocommit=True)


def save_quotes(quotes: list[tuple]) -> None:
    with db_connection.cursor() as cursor:
        cursor.executemany(
            """
            INSERT INTO quotes(close, high, interval, low, open, timestamp, symbol, volume) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)                
            ON CONFLICT DO NOTHING
            """, quotes)

def get_latest_quote_timestamp(symbol, interval) -> float:
    with db_connection.cursor() as cursor:
        cursor.execute("SELECT timestamp FROM quotes WHERE symbol = %s and interval = %s "
                       "ORDER BY timestamp DESC LIMIT 1", (symbol, interval))
        timestamp = cursor.fetchone()

        return 0 if timestamp is None else float(timestamp['timestamp'])


if __name__ == '__main__':
    interval = 15
    symbol = "SOL"

    session = HTTP(testnet=False)

    latest_timestamp_in_db = get_latest_quote_timestamp(symbol, interval)

    end_timestamp = latest_timestamp_in_db * 1000
    has_finished = False

    while not has_finished:
        kline_answer = session.get_kline(category="spot", symbol=f"{symbol}USDT", interval=interval, limit=1000,
                                         end=end_timestamp)

        bybit_quotes = kline_answer['result']['list']

        max_timestamp = max([quote[0] for quote in bybit_quotes])
        end_timestamp = min([quote[0] for quote in bybit_quotes])

        if len(bybit_quotes) < 2:
            has_finished = True
            print("\nQuotes fetched, exiting.")
            continue
        else:
            print(
                f"Fetched {len(bybit_quotes)} quotes for {symbol} "
                f"from:{datetime.fromtimestamp(int(end_timestamp) / 1000)} "
                f"to {datetime.fromtimestamp(int(max_timestamp) / 1000)}")

        quotes = [(quote[4], quote[2], interval, quote[3], quote[1], int(quote[0]) / 1000, symbol,
                   quote[5]) for quote in bybit_quotes]

        save_quotes(quotes)
