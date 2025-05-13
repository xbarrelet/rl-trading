import os
import argparse
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
import matplotlib.pyplot as plt

# Import your modules
from trading_env import TradingEnvironment
from training_agent import train_trading_agent, evaluate_trading_agent, plot_trading_results
from test_env import test_environment


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
    parser.add_argument('--position_size', type=float, default=0.03, help='Max position size as percentage of balance')
    parser.add_argument('--stop_loss', type=float, default=0.05, help='Stop loss percentage')
    parser.add_argument('--take_profit', type=float, default=0.1, help='Take profit percentage')
    parser.add_argument('--render', action='store_true', help='Render trading actions')

    args = parser.parse_args()

    # Import your data loading functions
    from data_preparation import load_data, add_indicators_to_df

    # Load and prepare data
    print(f"Loading data for {args.symbol} at {args.timeperiod}min interval...")
    df = load_data(symbol=args.symbol, timeperiod=args.timeperiod)
    add_indicators_to_df(df, args.timeperiod)
    df.dropna(inplace=True)

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

        # Create environment with specified parameters
        train_env = lambda: TradingEnvironment(
            train_df,
            initial_balance=args.initial_balance,
            position_size_percent=args.position_size,
            stop_loss_percent=args.stop_loss,
            take_profit_percent=args.take_profit,
            reward_scaling=1.0
        )

        # Train the agent
        print("Training agent...")
        model = train_trading_agent(train_df, eval_df, total_timesteps=args.timesteps)

        # Evaluate the agent on test data
        print("Evaluating agent on test data...")
        results = evaluate_trading_agent(model, test_df, render=args.render)

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

        # Evaluate the agent
        print(f"Evaluating agent on {len(test_df)} data points...")
        results = evaluate_trading_agent(model, test_df, render=args.render)

        # Plot results
        print("Plotting results...")
        plot_trading_results(results, args.symbol)


if __name__ == "__main__":
    main()