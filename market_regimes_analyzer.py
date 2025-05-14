import matplotlib.pyplot as plt

from data_preparation import load_data


def analyze_market_regimes(symbol='SOL', timeperiod=15):
    # Load the data
    df = load_data(symbol=symbol, timeperiod=timeperiod)

    # Count regimes
    regime_counts = df['market_regime'].value_counts().sort_index()
    print(f"Market Regime Distribution:")
    for regime, count in regime_counts.items():
        regime_name = {-2: "Strong Downtrend", -1: "Downtrend",
                       0: "Sideways", 1: "Uptrend", 2: "Strong Uptrend"}.get(regime, "Unknown")
        percent = count / len(df) * 100
        print(f"{regime_name} ({regime}): {count} samples ({percent:.2f}%)")

    # Plot price with regime coloring
    plt.figure(figsize=(14, 8))

    # Plot close price
    ax1 = plt.subplot(211)
    plt.plot(df['close'], color='black', alpha=0.6)
    plt.title(f"{symbol} Price with Market Regime Classification")
    plt.ylabel("Price")

    # Color the background based on market regime
    colors = {-2: 'darkred', -1: 'lightcoral', 0: 'lightgray',
              1: 'lightgreen', 2: 'darkgreen'}

    for regime in sorted(df['market_regime'].unique()):
        if regime not in colors:
            continue
        mask = df['market_regime'] == regime
        plt.fill_between(df.index, df['close'].min(), df['close'].max(),
                         where=mask, color=colors[regime], alpha=0.3)

    # Plot SMA difference
    ax2 = plt.subplot(212, sharex=ax1)
    plt.plot(df['SMA_diff_pct'], color='blue')
    plt.axhline(y=0.5, color='green', linestyle='--', alpha=0.7)
    plt.axhline(y=-0.5, color='red', linestyle='--', alpha=0.7)
    plt.axhline(y=1.0, color='darkgreen', linestyle='--', alpha=0.7)
    plt.axhline(y=-1.0, color='darkred', linestyle='--', alpha=0.7)
    plt.title("SMA Difference (%) with Regime Thresholds")
    plt.ylabel("SMA Diff %")

    plt.tight_layout()
    plt.savefig(f"{symbol}_market_regimes.png")
    plt.show()


if __name__ == "__main__":
    analyze_market_regimes()
