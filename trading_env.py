import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class TradingEnvironment(gym.Env):
    """Custom Trading Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df, initial_balance=100000, transaction_fee_percent=0.001,
                 position_size_percent=0.03, stop_loss_percent=0.05,
                 take_profit_percent=0.1, reward_scaling=1.0):
        super(TradingEnvironment, self).__init__()

        # Market data
        self.df = df
        self.features = self._extract_features()

        # Trading parameters
        self.initial_balance = initial_balance
        self.transaction_fee_percent = transaction_fee_percent
        self.position_size_percent = position_size_percent  # Max 3% per trade
        self.stop_loss_percent = stop_loss_percent  # 5% stop loss
        self.take_profit_percent = take_profit_percent  # 10% take profit
        self.reward_scaling = reward_scaling  # Scale rewards to help learning

        # Spaces - Extended action space for better granularity
        # 0: Hold, 1: Buy 1%, 2: Buy 2%, 3: Buy 3%, 4: Sell All
        self.action_space = spaces.Discrete(5)

        # Observation space: market features + position info
        feature_dim = self.features.shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(feature_dim + 4,),  # +4 for balance, position, cost_basis, unrealized_pnl
            dtype=np.float32
        )

        # Episode variables
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0
        self.cost_basis = 0
        self.total_reward = 0
        self.history = []
        self.last_trade_price = 0

        # Max steps
        self.max_steps = len(df) - 1

    def _extract_features(self):
        """Extract normalized features for the RL state"""
        # Select features from the dataframe
        feature_columns = [
            'close', 'SMA_10', 'EMA_10', 'SMA_20', 'EMA_20', 'SMA_50', 'EMA_50',
            'MACD', 'MACD_signal', 'MACD_hist', 'RSI', 'ROC', 'MOM',
            'upper_bb', 'middle_bb', 'lower_bb', 'TRIX', 'HT_TRENDLINE',
            'price_change_pct', 'log_return', 'RSI_MA_14', 'MACD_MA_14'
        ]

        # Extract features
        features_df = self.df[feature_columns]

        # Normalize features using min-max scaling
        normalized_df = (features_df - features_df.min()) / (features_df.max() - features_df.min() + 1e-8)

        # Fill NaN values
        normalized_df.fillna(0, inplace=True)

        return normalized_df.values

    def _get_observation(self):
        """Return the current observation"""
        # Market features
        features = self.features[self.current_step]

        # Calculate unrealized profit/loss if we have a position
        current_price = self.df.iloc[self.current_step]['close']
        unrealized_pnl = 0
        if self.position > 0:
            unrealized_pnl = (current_price - self.cost_basis) * self.position

        # Portfolio state
        portfolio_state = np.array([
            self.balance / self.initial_balance,  # Normalized balance
            self.position * current_price / self.initial_balance,  # Position size relative to initial balance
            self.cost_basis / current_price if self.position > 0 else 0,  # Normalized cost basis
            unrealized_pnl / self.initial_balance  # Normalized unrealized P&L
        ])

        # Combine market features and portfolio state
        obs = np.concatenate([features, portfolio_state])
        return obs.astype(np.float32)

    def _take_action(self, action):
        """Execute the action in the environment"""
        current_price = self.df.iloc[self.current_step]['close']
        reward = 0
        done = False
        info = {}

        # Check for stop loss or take profit if we have a position
        if self.position > 0:
            # Calculate profit/loss percentage
            unrealized_pnl_pct = (current_price - self.cost_basis) / self.cost_basis

            # Stop loss hit
            if unrealized_pnl_pct <= -self.stop_loss_percent:
                # Sell everything
                sale_amount = self.position * current_price
                fee = sale_amount * self.transaction_fee_percent
                sale_amount_after_fee = sale_amount - fee

                # Calculate P&L
                pl = sale_amount_after_fee - (self.position * self.cost_basis)
                # Apply negative reward for stop loss
                reward = pl * self.reward_scaling * 0.5  # Reduced reward due to stop loss

                # Update portfolio
                self.balance += sale_amount_after_fee
                self.position = 0
                self.cost_basis = 0

                info['stop_loss'] = True

            # Take profit hit
            elif unrealized_pnl_pct >= self.take_profit_percent:
                # Sell everything
                sale_amount = self.position * current_price
                fee = sale_amount * self.transaction_fee_percent
                sale_amount_after_fee = sale_amount - fee

                # Calculate P&L
                pl = sale_amount_after_fee - (self.position * self.cost_basis)
                # Apply extra reward for take profit
                reward = pl * self.reward_scaling * 1.5  # Enhanced reward for take profit

                # Update portfolio
                self.balance += sale_amount_after_fee
                self.position = 0
                self.cost_basis = 0

                info['take_profit'] = True

        # Hold
        if action == 0:
            # Small negative reward for holding to encourage action
            reward -= 0.0001 * self.reward_scaling

        # Buy actions (1, 2, 3 = Buy 1%, 2%, 3% of balance)
        elif action in [1, 2, 3] and self.balance > 0:
            # Calculate position size as percentage of initial balance
            size_percent = action * self.position_size_percent / 3.0  # Scale 1,2,3 to 1%,2%,3%
            buy_amount = min(self.balance, self.initial_balance * size_percent)

            if buy_amount > 0:
                # Calculate max tokens to buy with the position size
                tokens_to_buy = buy_amount / current_price

                # Apply transaction fee
                fee = tokens_to_buy * current_price * self.transaction_fee_percent
                tokens_after_fee = (buy_amount - fee) / current_price

                # Update portfolio
                if self.position > 0:
                    # Update cost basis (weighted average)
                    self.cost_basis = ((self.position * self.cost_basis) + (tokens_after_fee * current_price)) / (
                                self.position + tokens_after_fee)
                else:
                    self.cost_basis = current_price

                self.position += tokens_after_fee
                self.balance -= buy_amount
                self.last_trade_price = current_price

                # Small positive reward for taking action
                reward += 0.0001 * self.reward_scaling
                info['action_taken'] = 'buy'

        # Sell
        elif action == 4 and self.position > 0:
            # Calculate sale proceeds
            sale_amount = self.position * current_price

            # Apply transaction fee
            fee = sale_amount * self.transaction_fee_percent
            sale_amount_after_fee = sale_amount - fee

            # Calculate P&L
            pl = sale_amount_after_fee - (self.position * self.cost_basis)
            reward = pl * self.reward_scaling

            # Update portfolio
            self.balance += sale_amount_after_fee
            self.position = 0
            self.cost_basis = 0

            info['action_taken'] = 'sell'
            info['profit_loss'] = pl

        # Apply a small time-based penalty to encourage efficient trading
        reward -= 0.00005 * self.reward_scaling

        # Calculate portfolio value
        portfolio_value = self.balance + (self.position * current_price)

        # Record history
        self.history.append({
            'step': self.current_step,
            'timestamp': self.df.iloc[self.current_step]['timestamp'],
            'price': current_price,
            'action': ['HOLD', 'BUY_1%', 'BUY_2%', 'BUY_3%', 'SELL'][action],
            'position': self.position,
            'balance': self.balance,
            'portfolio_value': portfolio_value,
            'reward': reward,
            'cost_basis': self.cost_basis
        })

        return reward, done, info

    def step(self, action):
        """Take action and observe next state and reward"""
        # Execute action
        reward, done, info = self._take_action(action)

        # Move to next step
        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True

        # Get next observation
        observation = self._get_observation()

        # Accumulate reward
        self.total_reward += reward

        # Update info
        info['portfolio_value'] = self.balance + (self.position * self.df.iloc[self.current_step]['close'])
        info['total_reward'] = self.total_reward

        return observation, reward, done, False, info

    def reset(self, seed=None, options=None):
        """Reset the environment for a new episode"""
        # Reset episode variables
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.cost_basis = 0
        self.total_reward = 0
        self.history = []

        # Get initial observation
        observation = self._get_observation()

        # Optional info dictionary
        info = {}

        return observation, info

    def render(self, mode='human'):
        """Render the environment to the screen"""
        if mode == 'human':
            step = self.current_step - 1 if self.current_step > 0 else 0
            record = self.history[step] if self.history else {
                'step': self.current_step,
                'price': self.df.iloc[self.current_step]['close'],
                'action': 'NONE',
                'position': self.position,
                'balance': self.balance,
                'portfolio_value': self.balance
            }

            print(f"Step: {record['step']}")
            print(f"Timestamp: {record['timestamp']}")
            print(f"Price: ${record['price']:.2f}")
            print(f"Action: {record['action']}")
            print(f"Position: {record['position']:.6f}")
            print(f"Balance: ${record['balance']:.2f}")
            print(f"Portfolio Value: ${record['portfolio_value']:.2f}")
            print(f"Reward: {record.get('reward', 0):.2f}")
            print("-" * 50)

    def close(self):
        """Close the environment"""
        pass

    def get_trading_results(self):
        """Return trading history as DataFrame"""
        return pd.DataFrame(self.history)