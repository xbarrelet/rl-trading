import optuna
import pandas as pd
from training_agent import train_enhanced_trading_agent, adaptive_evaluate_trading_agent


def optimize_trading_agent(trial, train_df, eval_df, test_df, initial_balance=100000):
    """Objective function for Optuna optimization"""

    # Define the hyperparameters to optimize
    timesteps = trial.suggest_int('timesteps', 50000, 400000, step=50000)
    dynamic_stop_loss = trial.suggest_categorical('dynamic_stop_loss', [True, False])
    reward_scaling = trial.suggest_float('reward_scaling', 0.5, 2.0, step=0.25)
    lookback_window = trial.suggest_int('lookback_window', 10, 30, step=5)

    # Train the model with the suggested hyperparameters
    model = train_enhanced_trading_agent(
        train_df,
        eval_df,
        total_timesteps=timesteps,
        dynamic_stop_loss=dynamic_stop_loss,
        reward_scaling=reward_scaling,
        lookback_window=lookback_window
    )

    # Evaluate the model on test data
    results = adaptive_evaluate_trading_agent(
        model,
        test_df,
        render=False,
        dynamic_stop_loss=dynamic_stop_loss
    )

    # Calculate performance metrics
    if isinstance(results, pd.DataFrame) and not results.empty:
        final_value = results['portfolio_value'].iloc[-1]
        returns = (final_value - initial_balance) / initial_balance

        # Get win rate and max drawdown from environment
        win_rate = model.env.get_attr('profitable_trades')[0] / max(1, model.env.get_attr('trade_count')[0])
        max_drawdown = model.env.get_attr('max_drawdown')[0]

        # Objective: maximize returns while considering risk
        # Higher win rate, higher returns, and lower drawdown are better
        objective = returns * (1 + win_rate) * (1 - max_drawdown)
        return objective

    return -1.0  # Return a poor score if evaluation fails


def run_hyperparameter_optimization(symbol='SOL', timeperiod=30, n_trials=10):
    """Run hyperparameter optimization"""
    from data_preparation import load_data

    # Load and prepare data
    print(f"Loading data for {symbol} at {timeperiod}min interval...")
    df = load_data(symbol=symbol, timeperiod=timeperiod)

    # Split data for training, validation, and testing
    train_size = int(len(df) * 0.6)
    eval_size = int(len(df) * 0.2)

    train_df = df.iloc[:train_size].reset_index(drop=True)
    eval_df = df.iloc[train_size:train_size + eval_size].reset_index(drop=True)
    test_df = df.iloc[train_size + eval_size:].reset_index(drop=True)

    # Create Optuna study
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: optimize_trading_agent(trial, train_df, eval_df, test_df),
                   n_trials=n_trials,
                   n_jobs=-1)

    # Print results
    print("Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    print(f"Best score: {study.best_value}")

    return study.best_params


if __name__ == "__main__":
    # TODO: Increase trials to 50-100 for a better search
    best_params = run_hyperparameter_optimization(n_trials=1)