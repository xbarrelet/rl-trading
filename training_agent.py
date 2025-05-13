import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import pandas as pd

from trading_env import TradingEnvironment


class TradingCallback(BaseCallback):
    """Custom callback for monitoring and saving best model during training"""

    def __init__(self, eval_env, check_freq=1000, save_path='./models', verbose=1):
        super(TradingCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.check_freq = check_freq
        self.save_path = save_path
        self.best_mean_reward = -np.inf

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            # Run evaluation
            mean_reward = self._evaluate_agent()

            # Save the model if it's the best so far
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                print(f"Saving new best model with mean reward: {mean_reward:.2f}")
                self.model.save(os.path.join(self.save_path, f'best_model_{self.n_calls}'))

            print(
                f"Step: {self.n_calls}, Mean reward: {mean_reward:.2f}, Best mean reward: {self.best_mean_reward:.2f}")

        return True

    def _evaluate_agent(self):
        """Evaluate the agent on the evaluation environment"""
        obs, _ = self.eval_env.reset()
        done = False
        total_rewards = 0

        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = self.eval_env.step(action)
            total_rewards += reward

        return total_rewards


def train_trading_agent(train_df, eval_df=None, total_timesteps=100000, initial_balance=100000,
                        position_size=0.03, stop_loss=0.05, take_profit=0.1, reward_scaling=1.0):
    """Train a trading agent using PPO algorithm"""
    # If no eval data is provided, use a portion of the training data
    if eval_df is None:
        split_idx = int(len(train_df) * 0.8)
        eval_df = train_df.iloc[split_idx:].reset_index(drop=True)
        train_df = train_df.iloc[:split_idx].reset_index(drop=True)

    # Create environments with risk management parameters
    train_env = DummyVecEnv([
        lambda: TradingEnvironment(
            train_df,
            initial_balance=initial_balance,
            position_size_percent=position_size,
            stop_loss_percent=stop_loss,
            take_profit_percent=take_profit,
            reward_scaling=reward_scaling
        )
    ])

    eval_env = TradingEnvironment(
        eval_df,
        initial_balance=initial_balance,
        position_size_percent=position_size,
        stop_loss_percent=stop_loss,
        take_profit_percent=take_profit,
        reward_scaling=reward_scaling
    )

    # Create the callback
    os.makedirs('./models', exist_ok=True)
    callback = TradingCallback(eval_env, check_freq=5000, save_path='./models')

    # Create the agent
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=1e-4,  # Lower learning rate for stability
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,  # Higher entropy coefficient to encourage exploration
        verbose=1,
        tensorboard_log="./tensorboard_logs/"
    )

    # Train the agent
    model.learn(total_timesteps=total_timesteps, callback=callback)

    # Save the final model
    model.save("./models/final_model")

    return model


def evaluate_trading_agent(model, test_df, render=False):
    """Evaluate a trained trading agent on test data"""
    # Create test environment
    env = TradingEnvironment(test_df)

    # Reset the environment
    obs, _ = env.reset()
    done = False

    # Run evaluation
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)

        if render:
            env.render()

    # Get trading results
    results = env.get_trading_results()

    # Calculate performance metrics
    initial_value = env.initial_balance
    final_value = results.iloc[-1]['portfolio_value']
    returns = (final_value - initial_value) / initial_value * 100

    # Print results
    print(f"Initial Portfolio Value: ${initial_value:.2f}")
    print(f"Final Portfolio Value: ${final_value:.2f}")
    print(f"Returns: {returns:.2f}%")

    return results


def plot_trading_results(results, symbol):
    """Plot trading results"""
    # Create figure and axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    # Plot price and portfolio value
    ax1.plot(results['timestamp'], results['price'], label='Price')
    ax1.set_ylabel('Price')
    ax1.set_title(f'{symbol} Trading Results')
    ax1.legend(loc='upper left')

    # Add second y-axis for portfolio value
    ax1_twin = ax1.twinx()
    ax1_twin.plot(results['timestamp'], results['portfolio_value'], 'g-', label='Portfolio Value')
    ax1_twin.set_ylabel('Portfolio Value')
    ax1_twin.legend(loc='upper right')

    # Plot buy/sell actions
    buy_signals = results[results['action'] == 'BUY']
    sell_signals = results[results['action'] == 'SELL']

    ax2.plot(results['timestamp'], results['price'])
    ax2.scatter(buy_signals['timestamp'], buy_signals['price'], color='green', marker='^', s=100, label='Buy')
    ax2.scatter(sell_signals['timestamp'], sell_signals['price'], color='red', marker='v', s=100, label='Sell')
    ax2.set_ylabel('Price')
    ax2.set_xlabel('Time')
    ax2.legend()

    # Format x-axis
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(f"{symbol}_trading_results.png")
    plt.show()

    return fig


if __name__ == "__main__":
    from add_indicators import load_data, add_indicators_to_df

    # Load and prepare data
    symbol = "SOL"
    timeperiod = 15

    print(f"Loading data for {symbol} at {timeperiod}min interval...")
    df = load_data(symbol=symbol, timeperiod=timeperiod)
    add_indicators_to_df(df, timeperiod)
    df.dropna(inplace=True)

    # Split data for training, validation, and testing
    train_size = int(len(df) * 0.7)
    eval_size = int(len(df) * 0.15)

    train_df = df.iloc[:train_size].reset_index(drop=True)
    eval_df = df.iloc[train_size:train_size + eval_size].reset_index(drop=True)
    test_df = df.iloc[train_size + eval_size:].reset_index(drop=True)

    print(f"Training data: {len(train_df)} samples")
    print(f"Evaluation data: {len(eval_df)} samples")
    print(f"Testing data: {len(test_df)} samples")

    # Train the agent
    print("Training agent...")
    model = train_trading_agent(train_df, eval_df, total_timesteps=100000)

    # Evaluate the agent
    print("Evaluating agent...")
    results = evaluate_trading_agent(model, test_df, render=True)

    # Plot results
    print("Plotting results...")
    plot_trading_results(results, symbol)