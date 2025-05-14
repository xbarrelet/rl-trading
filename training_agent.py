import os

import numpy as np
import torch as th
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv

from trading_env import TradingEnvironment


class MarketAwareFeatureExtractor(BaseFeaturesExtractor):
    """
    Feature extractor that has separate processing pathways for market data and agent state.
    This allows the network to better understand market regimes and agent positioning.
    """

    def __init__(self, observation_space, features_dim=128):
        super(MarketAwareFeatureExtractor, self).__init__(observation_space, features_dim)

        total_features = observation_space.shape[0]
        # Assume last 6 features are agent state, rest are market features
        market_features = total_features - 6
        agent_features = 6  # balance, position, cost_basis, unrealized_pnl, regime, volatility

        # Market data pathway - deeper for better pattern recognition
        self.market_net = nn.Sequential(
            nn.Linear(market_features, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        # Agent state pathway - simpler
        self.agent_net = nn.Sequential(
            nn.Linear(agent_features, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        # Combine pathways
        self.combine = nn.Sequential(
            nn.Linear(64 + 32, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        # Split observation into market data and agent state
        market_data = observations[:, :-6]
        agent_state = observations[:, -6:]

        market_features = self.market_net(market_data)
        agent_features = self.agent_net(agent_state)

        # Concatenate both feature vectors
        combined = th.cat([market_features, agent_features], dim=1)
        return self.combine(combined)


class RegimeDetectionCallback(BaseCallback):
    """
    Callback that logs market regime statistics during training
    to help understand how the agent behaves in different regimes
    """

    def __init__(self, eval_env, log_freq=1000, verbose=1):
        super(RegimeDetectionCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.log_freq = log_freq

        # Store metrics by regime
        self.regime_actions = {-2: [], -1: [], 0: [], 1: [], 2: []}
        self.regime_rewards = {-2: [], -1: [], 0: [], 1: [], 2: []}
        self.regime_count = {-2: 0, -1: 0, 0: 0, 1: 0, 2: 0}

    def _on_step(self):
        if self.n_calls % self.log_freq == 0:
            # Evaluate the agent with regime tracking
            self._evaluate_with_regimes()

            # Print regime statistics
            print("\n===== Market Regime Statistics =====")
            for regime in [-2, -1, 0, 1, 2]:
                regime_name = "Strong Downtrend" if regime == -2 \
                    else "Downtrend" if regime == -1 \
                    else "Sideways" if regime == 0 \
                    else "Uptrend" if regime == 1 \
                    else "Strong Uptrend"

                count = self.regime_count[regime]

                if count > 0:
                    avg_reward = np.mean(self.regime_rewards[regime])
                    action_dist = np.bincount(self.regime_actions[regime], minlength=9) / len(
                        self.regime_actions[regime])

                    print(f"{regime_name}: {count} steps, Avg Reward: {avg_reward:.4f}")
                    print(
                        f"Action Distribution: Hold: {action_dist[0]:.2f}, Buy: {np.sum(action_dist[1:5]):.2f}, Sell: {np.sum(action_dist[5:]):.2f}")
                else:
                    print(f"{regime_name}: No data")
            print("=====================================\n")

            # Reset for next evaluation period
            self.regime_actions = {-2: [], -1: [], 0: [], 1: [], 2: []}
            self.regime_rewards = {-2: [], -1: [], 0: [], 1: [], 2: []}
            self.regime_count = {-2: 0, -1: 0, 0: 0, 1: 0, 2: 0}

        return True

    def _evaluate_with_regimes(self):
        """Evaluate agent and track performance by market regime"""
        obs, _ = self.eval_env.reset()
        done = False

        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            next_obs, reward, done, _, info = self.eval_env.step(action)

            # Extract market regime from info
            if 'market_regime' in info:
                regime = info['market_regime']

                # Record statistics
                self.regime_actions[regime].append(action)
                self.regime_rewards[regime].append(reward)
                self.regime_count[regime] += 1

            obs = next_obs


class TradingMetricsCallback(BaseCallback):
    """
    Advanced callback for tracking trading metrics during training
    such as Sharpe ratio, max drawdown, win rate, etc.
    """

    def __init__(self, eval_env, check_freq=1000, save_path='./models', verbose=1):
        super(TradingMetricsCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.check_freq = check_freq
        self.save_path = save_path
        self.best_sharpe = -np.inf
        self.best_reward = -np.inf

        # Create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            # Run evaluation
            metrics = self._evaluate_agent()

            # Save the model if it's the best by Sharpe ratio so far
            if metrics['sharpe_ratio'] > self.best_sharpe:
                self.best_sharpe = metrics['sharpe_ratio']
                self.model.save(os.path.join(self.save_path, f'best_sharpe_model_{self.n_calls}'))
                print(f"New best model saved with Sharpe ratio: {self.best_sharpe:.2f}")

            # Also save best by total reward
            if metrics['total_reward'] > self.best_reward:
                self.best_reward = metrics['total_reward']
                self.model.save(os.path.join(self.save_path, f'best_reward_model_{self.n_calls}'))

            # Print metrics
            print(f"\nStep: {self.n_calls}")
            print(f"Total Reward: {metrics['total_reward']:.2f}")
            print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            print(f"Win Rate: {metrics['win_rate'] * 100:.1f}%")
            print(f"Max Drawdown: {metrics['max_drawdown'] * 100:.1f}%")
            print(f"Profit Factor: {metrics['profit_factor']:.2f}")
            print(f"Trades: {metrics['trade_count']}")
            print(f"Final Value: ${metrics['final_value']:.2f}")

        return True

    def _evaluate_agent(self):
        """Evaluate the agent with detailed metrics"""
        obs, _ = self.eval_env.reset()
        done = False
        daily_returns = []

        # Track trades manually
        profit_trades = []
        loss_trades = []

        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, _, info = self.eval_env.step(action)

            # Capture profit/loss info when available
            if 'profit_loss' in info:
                pl = info['profit_loss']
                if pl > 0:
                    profit_trades.append(pl)
                elif pl < 0:
                    loss_trades.append(pl)

            if 'daily_return' in info:
                daily_returns.append(info['daily_return'])

        # Get trading results
        results = self.eval_env.unwrapped.get_trading_results()

        # Calculate metrics
        metrics = {
            'total_reward': self.eval_env.unwrapped.total_reward,
            'final_value': results.iloc[-1]['portfolio_value'] if not results.empty else 0,
            'win_rate': self.eval_env.unwrapped.profitable_trades / max(1, self.eval_env.unwrapped.trade_count),
            'max_drawdown': self.eval_env.unwrapped.max_drawdown,
            'trade_count': self.eval_env.unwrapped.trade_count,
        }

        # Calculate Sharpe ratio if we have daily returns
        if daily_returns and len(daily_returns) > 1:
            metrics['sharpe_ratio'] = np.mean(daily_returns) / (np.std(daily_returns) + 1e-6) * np.sqrt(252)
        else:
            metrics['sharpe_ratio'] = 0

        # Calculate profit factor using our manually tracked trades
        if loss_trades and sum(map(abs, loss_trades)) > 0:
            metrics['profit_factor'] = sum(profit_trades) / sum(map(abs, loss_trades))
        else:
            metrics['profit_factor'] = float('inf') if profit_trades else 0

        return metrics


def create_enhanced_ppo_model(env, tensorboard_log="./tensorboard_logs/"):
    """Create an enhanced PPO model with custom policy network"""

    policy_kwargs = {
        "features_extractor_class": MarketAwareFeatureExtractor,
        "features_extractor_kwargs": {"features_dim": 128},
        "net_arch": [
            dict(pi=[64, 64], vf=[64, 64])  # Separate networks for policy and value function
        ],
        "activation_fn": th.nn.ReLU
    }

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=0.2,
        normalize_advantage=True,
        ent_coef=0.01,  # Slightly lower than before for more exploitation
        vf_coef=0.5,  # Value function coefficient
        max_grad_norm=0.5,
        use_sde=False,  # Parameter-space noise for exploration
        sde_sample_freq=-1,
        target_kl=0.015,  # Early stopping on KL divergence
        tensorboard_log=tensorboard_log,
        policy_kwargs=policy_kwargs,
        verbose=1
    )

    return model


def train_enhanced_trading_agent(train_df, eval_df=None, total_timesteps=100000,
                                 initial_balance=100000, transaction_fee_percent=0.001,
                                 base_position_size_percent=0.02, max_position_size_percent=0.1,
                                 dynamic_stop_loss=True, reward_scaling=1.0, lookback_window=20):
    """Train an enhanced trading agent with improved features and callbacks"""

    # If no eval data is provided, use a portion of the training data
    if eval_df is None:
        split_idx = int(len(train_df) * 0.8)
        eval_df = train_df.iloc[split_idx:].reset_index(drop=True)
        train_df = train_df.iloc[:split_idx].reset_index(drop=True)

    # Create a directory for logs and models
    os.makedirs('./models', exist_ok=True)
    os.makedirs('./tensorboard_logs', exist_ok=True)

    # Set up training environment
    train_env = DummyVecEnv([
        lambda: Monitor(TradingEnvironment(
            train_df,
            initial_balance=initial_balance,
            transaction_fee_percent=transaction_fee_percent,
            base_position_size_percent=base_position_size_percent,
            max_position_size_percent=max_position_size_percent,
            dynamic_stop_loss=dynamic_stop_loss,
            reward_scaling=reward_scaling,
            lookback_window=lookback_window
        ))
    ])

    # Set up evaluation environment
    eval_env = Monitor(TradingEnvironment(
        eval_df,
        initial_balance=initial_balance,
        transaction_fee_percent=transaction_fee_percent,
        base_position_size_percent=base_position_size_percent,
        max_position_size_percent=max_position_size_percent,
        dynamic_stop_loss=dynamic_stop_loss,
        reward_scaling=reward_scaling,
        lookback_window=lookback_window
    ))

    # Create callbacks
    regime_callback = RegimeDetectionCallback(eval_env, log_freq=10000)
    metrics_callback = TradingMetricsCallback(eval_env, check_freq=5000, save_path='./models')
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./models/',
        log_path='./logs/',
        eval_freq=5000,
        deterministic=True,
        render=False
    )

    # Combine callbacks
    callbacks = [regime_callback, metrics_callback, eval_callback]

    # Create and train the model
    model = create_enhanced_ppo_model(train_env)
    model.learn(total_timesteps=total_timesteps, callback=callbacks)

    # Save the final model
    model.save("./models/final_model")

    return model


def adaptive_evaluate_trading_agent(model, test_df, render=False, initial_balance=100000,
                                    transaction_fee_percent=0.001, dynamic_stop_loss=True):
    """
    Evaluate a trained trading agent on test data with adaptive position sizing and
    risk management based on market regimes and volatility
    """
    # Create test environment
    env = TradingEnvironment(
        test_df,
        initial_balance=initial_balance,
        transaction_fee_percent=transaction_fee_percent,
        dynamic_stop_loss=dynamic_stop_loss
    )

    # Reset the environment
    obs, _ = env.reset()
    done = False

    # For analyzing performance by market regime
    regime_metrics = {
        -1: {'returns': [], 'actions': [], 'rewards': []},  # Downtrend
        0: {'returns': [], 'actions': [], 'rewards': []},  # Sideways
        1: {'returns': [], 'actions': [], 'rewards': []}  # Uptrend
    }

    # Run evaluation
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)

        if render:
            env.render()

        # Track metrics by regime if available
        if 'market_regime' in info:
            regime = info['market_regime']
            if regime in regime_metrics:
                regime_metrics[regime]['rewards'].append(reward)
                regime_metrics[regime]['actions'].append(action)

                # Calculate returns if portfolio value available
                if 'portfolio_value' in info and env.history and len(env.history) > 1:
                    prev_value = env.history[-2]['portfolio_value'] if len(env.history) > 1 else initial_balance
                    current_value = info['portfolio_value']
                    pct_return = (current_value - prev_value) / prev_value if prev_value > 0 else 0
                    regime_metrics[regime]['returns'].append(pct_return)

    # Get trading results
    results = env.get_trading_results()

    # Calculate performance metrics
    initial_value = env.initial_balance
    final_value = results.iloc[-1]['portfolio_value'] if not results.empty else 0
    returns = (final_value - initial_value) / initial_value * 100

    # Print overall results
    print("\n===== EVALUATION RESULTS =====")
    print(f"Initial Portfolio Value: ${initial_value:.2f}")
    print(f"Final Portfolio Value: ${final_value:.2f}")
    print(f"Returns: {returns:.2f}%")
    print(f"Number of trades: {env.trade_count}")
    print(f"Win rate: {env.profitable_trades / max(1, env.trade_count) * 100:.2f}%")
    print(f"Maximum drawdown: {env.max_drawdown * 100:.2f}%")

    # Print regime-specific performance
    print("\n===== PERFORMANCE BY MARKET REGIME =====")
    for regime, data in regime_metrics.items():
        regime_name = "Downtrend" if regime == -1 else "Sideways" if regime == 0 else "Uptrend"
        if data['returns']:
            avg_return = np.mean(data['returns']) * 100
            avg_reward = np.mean(data['rewards'])

            # Action distribution
            actions = np.array(data['actions'])
            hold_pct = np.mean(actions == 0) * 100
            buy_pct = np.mean((actions >= 1) & (actions <= 4)) * 100
            sell_pct = np.mean(actions >= 5) * 100

            print(f"{regime_name}:")
            print(f"  Samples: {len(data['returns'])}")
            print(f"  Avg Return: {avg_return:.4f}%")
            print(f"  Avg Reward: {avg_reward:.4f}")
            print(f"  Actions: Hold {hold_pct:.1f}%, Buy {buy_pct:.1f}%, Sell {sell_pct:.1f}%")
        else:
            print(f"{regime_name}: No data")

    return results


if __name__ == "__main__":
    from data_preparation import load_data

    # Example usage
    symbol = "BTC"
    timeperiod = 60

    print(f"Loading data for {symbol} at {timeperiod}min interval...")
    df = load_data(symbol=symbol, timeperiod=timeperiod)

    # Split data
    train_size = int(len(df) * 0.6)
    eval_size = int(len(df) * 0.2)

    train_df = df.iloc[:train_size].reset_index(drop=True)
    eval_df = df.iloc[train_size:train_size + eval_size].reset_index(drop=True)
    test_df = df.iloc[train_size + eval_size:].reset_index(drop=True)

    # Train model
    model = train_enhanced_trading_agent(
        train_df, eval_df,
        total_timesteps=100000,
        reward_scaling=1.0,
        dynamic_stop_loss=True
    )

    # Evaluate
    results = adaptive_evaluate_trading_agent(
        model, test_df,
        render=True,
        dynamic_stop_loss=True
    )
