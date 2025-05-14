import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import PPO

from test_env import test_environment
from training_agent import train_enhanced_trading_agent, adaptive_evaluate_trading_agent


def plot_trading_results(results, symbol):
    """Plot trading results including portfolio value and actions"""
    if not isinstance(results, pd.DataFrame) or results.empty:
        print("No results to plot")
        return

    # Create directory for plots
    os.makedirs('plots', exist_ok=True)

    # Plot portfolio value over time
    plt.figure(figsize=(14, 7))
    plt.subplot(2, 1, 1)
    plt.plot(results['portfolio_value'], label='Portfolio Value')
    plt.title(f'Trading Results for {symbol}')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True)
    plt.legend()

    # Plot price and buy/sell actions
    plt.subplot(2, 1, 2)
    plt.plot(results['price'], label='Price', color='blue', alpha=0.6)

    # Plot buy and sell points
    for i, row in results.iterrows():
        if 'BUY' in row['action']:
            plt.scatter(i, row['price'], color='green', marker='^', alpha=0.7)
        elif 'SELL' in row['action']:
            plt.scatter(i, row['price'], color='red', marker='v', alpha=0.7)

    plt.xlabel('Time Steps')
    plt.ylabel('Price')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'plots/{symbol}_trading_results.png')
    plt.close()


def main():
    """Main function to run the trading bot"""
    parser = argparse.ArgumentParser(description='Reinforcement Learning Trading Bot')
    parser.add_argument('--symbol', type=str, default='SOL', help='Trading symbol')
    parser.add_argument('--timeperiod', type=int, default=15, help='Time period in minutes')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'evaluate'],
                        help='Operation mode')
    parser.add_argument('--model_path', type=str, default='./models/final_model',
                        help='Path to saved model for evaluation')
    parser.add_argument('--timesteps', type=int, default=200000, help='Total timesteps for training')
    parser.add_argument('--initial_balance', type=float, default=100000, help='Initial balance for trading')
    parser.add_argument('--base_position_size', type=float, default=0.02,
                        help='Base position size as percentage of balance')
    parser.add_argument('--max_position_size', type=float, default=0.03,
                        help='Maximum position size as percentage of balance')
    parser.add_argument('--dynamic_stop_loss', action='store_true', default=True,
                        help='Use dynamic stop loss based on market conditions')
    parser.add_argument('--transaction_fee', type=float, default=0.06,
                        help='Transaction fee as percentage')
    parser.add_argument('--reward_scaling', type=float, default=1.0,
                        help='Reward scaling factor')
    parser.add_argument('--lookback_window', type=int, default=20,
                        help='Lookback window for observations')
    parser.add_argument('--render', action='store_true', help='Render trading actions')

    args = parser.parse_args()

    # Import data preparation functions
    from data_preparation import load_data

    # Load and prepare data
    print(f"Loading data for {args.symbol} at {args.timeperiod}min interval...")
    df = load_data(symbol=args.symbol, timeperiod=args.timeperiod)

    if args.mode == 'train':
        # Split data for training, validation, and testing
        train_size = int(len(df) * 0.7)
        eval_size = int(len(df) * 0.15)

        train_df = df.iloc[:train_size].reset_index(drop=True)
        eval_df = df.iloc[train_size:train_size + eval_size].reset_index(drop=True)
        test_df = df.iloc[train_size + eval_size:].reset_index(drop=True)

        print(f"Training data: {len(train_df)} samples")
        print(f"Evaluation data: {len(eval_df)} samples")
        print(f"Testing data: {len(test_df)} samples")

        # Train the agent with updated parameters
        print("Training agent...")
        model = train_enhanced_trading_agent(
            train_df,
            eval_df,
            total_timesteps=args.timesteps,
            initial_balance=args.initial_balance,
            transaction_fee_percent=args.transaction_fee,
            base_position_size_percent=args.base_position_size,
            max_position_size_percent=args.max_position_size,
            dynamic_stop_loss=args.dynamic_stop_loss,
            reward_scaling=args.reward_scaling,
            lookback_window=args.lookback_window
        )

        # Evaluate the trained agent on test data
        print("Evaluating agent on test data...")
        results = adaptive_evaluate_trading_agent(
            model,
            test_df,
            render=args.render,
            initial_balance=args.initial_balance,
            transaction_fee_percent=args.transaction_fee,
            dynamic_stop_loss=args.dynamic_stop_loss
        )

        # Plot results
        print("Plotting results...")
        plot_trading_results(results, args.symbol)

    elif args.mode == 'test':
        # Use a portion of the data for testing the environment
        test_df = df.iloc[-500:].reset_index(drop=True)

        # Test the environment with random actions
        print(f"Testing environment with {len(test_df)} data points...")
        results = test_environment(test_df, num_episodes=3, render=args.render)

    elif args.mode == 'evaluate':
        # Load the trained model
        print(f"Loading model from {args.model_path}...")
        model = PPO.load(args.model_path)

        # Use test data for evaluation
        test_size = int(len(df) * 0.2)
        test_df = df.iloc[-test_size:].reset_index(drop=True)

        # Evaluate the agent with updated parameters
        print(f"Evaluating agent on {len(test_df)} data points...")
        results = adaptive_evaluate_trading_agent(
            model,
            test_df,
            render=args.render,
            initial_balance=args.initial_balance,
            transaction_fee_percent=args.transaction_fee,
            dynamic_stop_loss=args.dynamic_stop_loss
        )

        # Plot results
        print("Plotting results...")
        plot_trading_results(results, args.symbol)


if __name__ == "__main__":
    main()