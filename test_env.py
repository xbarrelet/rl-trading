import numpy as np
import matplotlib.pyplot as plt

from trading_env import TradingEnvironment


def test_environment(df, num_episodes=5, render=True):
    """Test the trading environment with random actions"""
    # Create environment
    env = TradingEnvironment(df)

    results = []

    # Run multiple episodes
    for episode in range(num_episodes):
        print(f"Episode {episode + 1}/{num_episodes}")

        # Reset environment
        observation, info = env.reset()
        done = False
        episode_reward = 0

        # Run episode
        while not done:
            # Take random action
            action = np.random.randint(0, 3)  # 0: Hold, 1: Buy, 2: Sell
            observation, reward, done, truncated, info = env.step(action)
            episode_reward += reward

            if render:
                env.render()

        # Get results
        episode_results = env.get_trading_results()

        # Record performance
        initial_value = env.initial_balance
        final_value = episode_results.iloc[-1]['portfolio_value']
        returns = (final_value - initial_value) / initial_value * 100

        results.append({
            'episode': episode,
            'initial_value': initial_value,
            'final_value': final_value,
            'returns': returns,
            'reward': episode_reward
        })

        print(
            f"Episode {episode + 1} - Initial: ${initial_value:.2f}, Final: ${final_value:.2f}, Returns: {returns:.2f}%, Reward: {episode_reward:.2f}")
        print("=" * 50)

    return results


if __name__ == "__main__":
    # Import your data loading functions
    from data_preparation import load_data, add_indicators_to_df

    # Load and prepare data
    symbol = "SOL"
    timeperiod = 15

    print(f"Loading data for {symbol} at {timeperiod}min interval...")
    df = load_data(symbol=symbol, timeperiod=timeperiod)
    add_indicators_to_df(df, timeperiod)
    df.dropna(inplace=True)

    # Use just a portion of the data for testing
    test_df = df.iloc[-500:].reset_index(drop=True)

    # Test the environment
    print(f"Testing environment with {len(test_df)} data points...")
    results = test_environment(test_df, num_episodes=3, render=True)

    # Plot results
    episodes = [r['episode'] for r in results]
    returns = [r['returns'] for r in results]

    plt.figure(figsize=(10, 5))
    plt.bar(episodes, returns)
    plt.title('Returns by Episode (Random Policy)')
    plt.xlabel('Episode')
    plt.ylabel('Returns (%)')
    plt.grid(True, alpha=0.3)

    plt.savefig("random_policy_results.png")
    plt.show()