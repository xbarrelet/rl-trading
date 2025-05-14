- Can I use algo strategies knowledge to improve the model somehow?
  - Like using T3 and other indicators?
  - ???
- Can I check what indicators is actually useful/used like with SHARP?


Key Improvement Areas
1. Feature Engineering & Market Context
Your current feature set is primarily technical indicators, but you could enhance it with:

Market regime identification - add features that help identify trending vs ranging markets
Volatility measures - incorporate ATR (Average True Range) or historical volatility
Volume indicators - volume often precedes price movements
Correlation with broader market - relationship with market indexes or sector performance

2. Reward Function Optimization
The current reward function might be improved by:

Better balancing of long-term vs short-term rewards
Adding a Sharpe ratio component to reward risk-adjusted returns
Penalizing drawdowns more explicitly
Considering asymmetric rewards (loss aversion)

3. Environment Design
Some adjustments to the trading environment:

Consider a continuous action space instead of discrete for more fine-grained position sizing
Implement a more sophisticated portfolio management approach
Add more realistic market frictions (slippage, variable fees based on volume)

4. Algorithm & Training Improvements
The PPO implementation could be enhanced by:

Hyperparameter tuning (particularly learning rate, batch size, and entropy coefficient)
Implementing experience replay with prioritized sampling
Testing other algorithms like SAC (Soft Actor-Critic) or TD3 (Twin Delayed DDPG)