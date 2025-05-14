import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


class TradingEnvironment(gym.Env):
    """Enhanced Trading Environment that adapts to market regimes and volatility"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df, initial_balance=100000, transaction_fee_percent=0.001,
                 base_position_size_percent=0.02, max_position_size_percent=0.1,
                 dynamic_stop_loss=True, reward_scaling=1.0,
                 lookback_window=20, consecutive_only=True):
        super(TradingEnvironment, self).__init__()

        # Market data
        self.df = df
        if consecutive_only and 'period' in df.columns:
            # Use only the largest consecutive period for training
            largest_period = df['period'].value_counts().idxmax()
            self.df = df[df['period'] == largest_period].reset_index(drop=True)
            print(f"Using largest consecutive period with {len(self.df)} samples")

        self.features = self._extract_features()
        self.lookback_window = lookback_window

        # Trading parameters
        self.initial_balance = initial_balance
        self.transaction_fee_percent = transaction_fee_percent
        self.base_position_size_percent = base_position_size_percent
        self.max_position_size_percent = max_position_size_percent
        self.dynamic_stop_loss = dynamic_stop_loss
        self.reward_scaling = reward_scaling

        # Extended action space:
        # 0: Hold
        # 1-4: Buy (1%, 2%, 5%, 10% of balance)
        # 5-8: Sell (25%, 50%, 75%, 100% of position)
        self.action_space = spaces.Discrete(9)

        # Observation space: market features + position info + market regime
        feature_dim = self.features.shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(feature_dim + 6,),  # +6 for balance, position, cost_basis, unrealized_pnl, regime, volatility
            dtype=np.float32
        )

        # Episode variables
        self.current_step = self.lookback_window  # Start after lookback window
        self.balance = initial_balance
        self.position = 0
        self.cost_basis = 0
        self.total_reward = 0
        self.history = []
        self.trade_count = 0
        self.profitable_trades = 0
        self.current_drawdown = 0
        self.max_drawdown = 0
        self.peak_portfolio_value = initial_balance

        # Stop loss and take profit will be dynamic based on ATR and market regime
        self.stop_loss_percent = 0.05  # Default, will be adjusted
        self.take_profit_percent = 0.1  # Default, will be adjusted

        # Max steps
        self.max_steps = len(self.df) - 1

    def _extract_features(self):
        """Extract normalized features for the RL state"""
        # Select features from the dataframe - now with enhanced features
        feature_columns = [
            # Primary price data
            'close', 'price_change_pct', 'log_return',

            # Trend indicators
            'SMA_10', 'EMA_10', 'SMA_20', 'EMA_20', 'SMA_50', 'EMA_50',
            'MACD', 'MACD_signal', 'MACD_hist',

            # Momentum indicators
            'RSI', 'ROC', 'MOM', 'RSI_MA_14', 'MACD_MA_14', 'CMO',

            # Volatility indicators
            'upper_bb', 'middle_bb', 'lower_bb', 'ATR_pct',
            'volatility_5', 'volatility_20', 'bb_width', 'volatility_change',

            # Market regime indicators
            'ADX', 'aroon_oscillator', 'HT_DCPERIOD', 'SMA_diff_pct',
        ]

        # Add volume indicators if they exist
        volume_indicators = ['OBV_norm', 'MFI', 'ADOSC', 'volume_ratio', 'PVT']
        for indicator in volume_indicators:
            if indicator in self.df.columns:
                feature_columns.append(indicator)

        # Extract features
        features_df = self.df[feature_columns]

        # Calculate normalized features using rolling window for better normalization
        window_size = 20
        normalized_df = pd.DataFrame()

        for col in features_df.columns:
            # Get the rolling mean and std for each feature
            rolling_mean = features_df[col].rolling(window=window_size).mean()
            rolling_std = features_df[col].rolling(window=window_size).std()

            # Normalize using the rolling stats (z-score normalization)
            normalized_df[col] = (features_df[col] - rolling_mean) / (rolling_std + 1e-8)

        # Fill NaN values that occur at the beginning due to rolling window
        normalized_df.fillna(0, inplace=True)

        return normalized_df.values

    def _adjust_risk_parameters(self):
        """Adjust risk parameters based on market regime and volatility"""
        current_idx = self.current_step

        # Get current market regime (-1 = downtrend, 0 = sideways, 1 = uptrend)
        market_regime = self.df.iloc[current_idx]['market_regime']

        # Get current volatility
        volatility = self.df.iloc[current_idx]['volatility_20']
        atr_pct = self.df.iloc[current_idx]['ATR_pct']

        # Get market strength
        adx = self.df.iloc[current_idx]['ADX']

        # Adjust stop loss based on ATR (tighter in low vol, wider in high vol)
        base_stop = 2.0  # Base multiplier for ATR
        if self.dynamic_stop_loss:
            # Use ATR to set stop loss, with adjustments for market regime
            if market_regime == 1:  # Uptrend
                self.stop_loss_percent = min(0.08, atr_pct * base_stop)  # More room in uptrend
                self.take_profit_percent = max(0.05, atr_pct * 3.0)  # Higher targets in uptrend
            elif market_regime == -1:  # Downtrend
                self.stop_loss_percent = min(0.05, atr_pct * base_stop)  # Tighter in downtrend
                self.take_profit_percent = max(0.08, atr_pct * 4.0)  # Higher reward needed in downtrend
            else:  # Sideways
                self.stop_loss_percent = min(0.04, atr_pct * base_stop)  # Tighter in sideways markets
                self.take_profit_percent = max(0.06, atr_pct * 2.5)  # Modest targets in sideways

        # Ensure minimum and maximum values
        self.stop_loss_percent = max(0.01, min(0.1, self.stop_loss_percent))
        self.take_profit_percent = max(0.02, min(0.2, self.take_profit_percent))

        return market_regime, volatility

    def _get_observation(self):
        """Return the current observation with lookback window consideration"""
        # Market features
        features = self.features[self.current_step]

        # Get market regime and volatility
        market_regime, volatility = self._adjust_risk_parameters()
        adx_strength = self.df.iloc[self.current_step]['ADX'] / 100.0  # Normalized ADX

        # Calculate unrealized profit/loss if we have a position
        current_price = self.df.iloc[self.current_step]['close']
        unrealized_pnl = 0
        unrealized_pnl_pct = 0
        if self.position > 0:
            unrealized_pnl = (current_price - self.cost_basis) * self.position
            unrealized_pnl_pct = (current_price - self.cost_basis) / self.cost_basis

        # Portfolio state
        portfolio_state = np.array([
            self.balance / self.initial_balance,  # Normalized balance
            self.position * current_price / self.initial_balance,  # Position size relative to initial balance
            self.cost_basis / current_price if self.position > 0 else 0,  # Normalized cost basis
            unrealized_pnl_pct,  # Percentage P&L of current position
            market_regime / 2.0,  # Normalized market regime (-0.5 to 0.5)
            adx_strength  # Trend strength
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

                # Apply negative reward for stop loss, but less severe than the actual loss
                # (to encourage the agent to take reasonable risks)
                pl = sale_amount_after_fee - (self.position * self.cost_basis)
                reward = pl * self.reward_scaling * 0.7  # Reduced penalty for stop loss

                # Update portfolio
                self.balance += sale_amount_after_fee
                self.position = 0
                self.cost_basis = 0
                self.trade_count += 1

                info['stop_loss'] = True
                info['profit_loss'] = pl

            # Take profit hit
            elif unrealized_pnl_pct >= self.take_profit_percent:
                # Sell everything
                sale_amount = self.position * current_price
                fee = sale_amount * self.transaction_fee_percent
                sale_amount_after_fee = sale_amount - fee

                # Apply enhanced reward for take profit
                pl = sale_amount_after_fee - (self.position * self.cost_basis)
                reward = pl * self.reward_scaling * 1.3  # Enhanced reward for take profit

                # Update portfolio
                self.balance += sale_amount_after_fee
                self.position = 0
                self.cost_basis = 0
                self.trade_count += 1
                self.profitable_trades += 1

                info['take_profit'] = True
                info['profit_loss'] = pl

        # Hold
        if action == 0:
            # Small negative reward for holding to encourage strategic action
            reward -= 0.0001 * self.reward_scaling

        # Buy actions with different position sizes
        elif action >= 1 and action <= 4 and self.balance > 0:
            # Map actions 1-4 to position sizes (1%, 2%, 5%, 10%)
            position_sizes = [0.01, 0.02, 0.05, 0.1]
            size_percent = position_sizes[action - 1]

            # Apply position size limit based on market conditions
            market_regime, _ = self._adjust_risk_parameters()

            # Limit position size in downtrends or high volatility
            if market_regime == -1:  # Downtrend
                size_percent = min(size_percent, self.base_position_size_percent)
            elif market_regime == 0:  # Sideways
                size_percent = min(size_percent, self.base_position_size_percent * 1.5)

            buy_amount = min(self.balance, self.initial_balance * size_percent)

            if buy_amount > 0:
                # Calculate tokens to buy
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

                # Small reward for taking action in the right market direction
                market_regime, _ = self._adjust_risk_parameters()
                if market_regime == 1:  # If buying in uptrend
                    reward += 0.0002 * self.reward_scaling
                elif market_regime == -1:  # If buying in downtrend (slight penalty)
                    reward -= 0.0001 * self.reward_scaling

                info['action_taken'] = f'buy_{size_percent * 100}%'

        # Sell actions with different position portions
        elif action >= 5 and action <= 8 and self.position > 0:
            # Map actions 5-8 to sell percentages (25%, 50%, 75%, 100%)
            sell_percentages = [0.25, 0.5, 0.75, 1.0]
            sell_percent = sell_percentages[action - 5]

            # Calculate tokens to sell
            tokens_to_sell = self.position * sell_percent

            # Calculate sale proceeds
            sale_amount = tokens_to_sell * current_price

            # Apply transaction fee
            fee = sale_amount * self.transaction_fee_percent
            sale_amount_after_fee = sale_amount - fee

            # Calculate P&L
            pl = sale_amount_after_fee - (tokens_to_sell * self.cost_basis)

            # Base reward on profit/loss
            reward = pl * self.reward_scaling

            # Add bonus for selling at the right time
            market_regime, _ = self._adjust_risk_parameters()
            if market_regime == -1 and pl > 0:  # Selling profitably in downtrend
                reward *= 1.2  # 20% bonus
            elif market_regime == 1 and pl < 0:  # Selling at a loss in uptrend
                reward *= 0.8  # 20% penalty

            # Update portfolio
            self.balance += sale_amount_after_fee
            self.position -= tokens_to_sell

            # Update trade statistics
            if sell_percent == 1.0:  # If selling entire position
                self.trade_count += 1
                if pl > 0:
                    self.profitable_trades += 1
                self.cost_basis = 0

            info['action_taken'] = f'sell_{sell_percent * 100}%'
            info['profit_loss'] = pl

        # Apply a small time-based penalty to encourage efficient trading
        reward -= 0.00005 * self.reward_scaling

        # Calculate portfolio value and drawdown
        portfolio_value = self.balance + (self.position * current_price)

        # Update drawdown tracking
        if portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = portfolio_value
            self.current_drawdown = 0
        else:
            self.current_drawdown = (self.peak_portfolio_value - portfolio_value) / self.peak_portfolio_value
            self.max_drawdown = max(self.max_drawdown, self.current_drawdown)

        # Record history
        self.history.append({
            'step': self.current_step,
            'timestamp': self.df.iloc[self.current_step]['timestamp'],
            'price': current_price,
            'action': ['HOLD', 'BUY_1%', 'BUY_2%', 'BUY_5%', 'BUY_10%',
                       'SELL_25%', 'SELL_50%', 'SELL_75%', 'SELL_100%'][action],
            'position': self.position,
            'balance': self.balance,
            'portfolio_value': portfolio_value,
            'reward': reward,
            'cost_basis': self.cost_basis,
            'market_regime': self.df.iloc[self.current_step]['market_regime'],
            'stop_loss': self.stop_loss_percent,
            'take_profit': self.take_profit_percent,
            'drawdown': self.current_drawdown
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

            # Force liquidation at the end of the episode
            if self.position > 0:
                current_price = self.df.iloc[self.current_step - 1]['close']
                sale_amount = self.position * current_price
                fee = sale_amount * self.transaction_fee_percent
                sale_amount_after_fee = sale_amount - fee

                # Add to balance
                self.balance += sale_amount_after_fee
                self.position = 0

        # Get next observation
        observation = self._get_observation()

        # Accumulate reward
        self.total_reward += reward

        # Update info
        current_price = self.df.iloc[min(self.current_step, self.max_steps - 1)]['close']
        portfolio_value = self.balance + (self.position * current_price)

        info['portfolio_value'] = portfolio_value
        info['total_reward'] = self.total_reward
        info['win_rate'] = self.profitable_trades / max(1, self.trade_count)
        info['max_drawdown'] = self.max_drawdown

        info['market_regime'] = self.df.iloc[min(self.current_step - 1, self.max_steps - 1)]['market_regime']

        return observation, reward, done, False, info

    def reset(self, seed=None, options=None):
        """Reset the environment for a new episode"""
        # Reset episode variables
        self.current_step = self.lookback_window  # Start after lookback window
        self.balance = self.initial_balance
        self.position = 0
        self.cost_basis = 0
        self.total_reward = 0
        self.history = []
        self.trade_count = 0
        self.profitable_trades = 0
        self.current_drawdown = 0
        self.max_drawdown = 0
        self.peak_portfolio_value = self.initial_balance

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
                'portfolio_value': self.balance,
                'market_regime': self.df.iloc[self.current_step]['market_regime'],
                'stop_loss': self.stop_loss_percent,
                'take_profit': self.take_profit_percent
            }

            print(f"Step: {record['step']}")
            print(f"Timestamp: {record['timestamp']}")
            print(f"Price: ${record['price']:.2f}")
            print(f"Action: {record['action']}")
            print(f"Position: {record['position']:.6f}")
            print(f"Balance: ${record['balance']:.2f}")
            print(f"Portfolio Value: ${record['portfolio_value']:.2f}")
            print(f"Market Regime: {record.get('market_regime', 0)}")
            print(f"Stop Loss: {record.get('stop_loss', 0):.2%}")
            print(f"Take Profit: {record.get('take_profit', 0):.2%}")
            print(f"Reward: {record.get('reward', 0):.4f}")
            print(f"Drawdown: {record.get('drawdown', 0):.2%}")
            print("-" * 50)

    def close(self):
        """Close the environment"""
        pass

    def get_trading_results(self):
        """Return trading history as DataFrame"""
        return pd.DataFrame(self.history)