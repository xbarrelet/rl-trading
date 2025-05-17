import pandas as pd
import argparse
import os
import itertools
import time
import logging
from datetime import datetime

from data_preparation import load_data
from training_agent import train_enhanced_trading_agent, adaptive_evaluate_trading_agent

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_grid_search(symbol='SOL', timeperiod=60, initial_balance=100000, transaction_fee=0.06):
    """Run grid search hyperparameter optimization for the trading agent using final value as metric"""

    # Ensure directories exist
    os.makedirs("grid_search_results", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Define parameter grid
    param_grid = {
        "timesteps": [50000],
        "dynamic_stop_loss": [True],
        "reward_scaling": [0.5, 1.0, 2.0],
        "lookback_window": [20]
    }

    # Calculate total combinations
    total_combinations = (
            len(param_grid["timesteps"]) *
            len(param_grid["dynamic_stop_loss"]) *
            len(param_grid["reward_scaling"]) *
            len(param_grid["lookback_window"])
    )

    logger.info(f"Starting grid search with {total_combinations} combinations")

    # Load and prepare data
    logger.info(f"Loading data for {symbol} at {timeperiod}min interval...")
    df = load_data(symbol=symbol, timeperiod=timeperiod)

    # Split data
    train_size = int(len(df) * 0.6)
    eval_size = int(len(df) * 0.2)

    train_df = df.iloc[:train_size].reset_index(drop=True)
    eval_df = df.iloc[train_size:train_size + eval_size].reset_index(drop=True)
    test_df = df.iloc[train_size + eval_size:].reset_index(drop=True)

    logger.info(f"Data split: Train={len(train_df)}, Eval={len(eval_df)}, Test={len(test_df)}")

    # Initialize results tracking
    results = []
    best_final_value = float('-inf')
    best_params = None

    # Generate all combinations
    param_combinations = list(itertools.product(
        param_grid["timesteps"],
        param_grid["dynamic_stop_loss"],
        param_grid["reward_scaling"],
        param_grid["lookback_window"]
    ))

    # Loop through combinations
    for i, (timesteps, dynamic_stop_loss, reward_scaling, lookback_window) in enumerate(param_combinations):
        trial_id = i + 1
        start_time = time.time()

        logger.info(f"Trial {trial_id}/{total_combinations} - Testing parameters: "
                    f"timesteps={timesteps}, "
                    f"dynamic_stop_loss={dynamic_stop_loss}, "
                    f"reward_scaling={reward_scaling}, "
                    f"lookback_window={lookback_window}")

        try:
            # Train model with hyperparameters
            model_path = f"./models/trial_{trial_id}"
            os.makedirs(model_path, exist_ok=True)

            model = train_enhanced_trading_agent(
                train_df=train_df,
                eval_df=eval_df,
                total_timesteps=timesteps,
                initial_balance=initial_balance,
                transaction_fee_percent=transaction_fee / 100,
                base_position_size_percent=0.02,
                max_position_size_percent=0.1,
                dynamic_stop_loss=dynamic_stop_loss,
                reward_scaling=reward_scaling,
                lookback_window=lookback_window
            )

            # Save model
            model.save(f"{model_path}/model")

            # Evaluate on test data
            results_df = adaptive_evaluate_trading_agent(
                model=model,
                test_df=test_df,
                render=False,
                initial_balance=initial_balance,
                transaction_fee_percent=transaction_fee / 100,
                dynamic_stop_loss=dynamic_stop_loss
            )

            # Extract metrics for scoring
            if results_df is not None and not results_df.empty:
                final_value = results_df.iloc[-1]['portfolio_value']
                max_drawdown = results_df['drawdown'].max() if 'drawdown' in results_df.columns else 0

                # Calculate ROI percentage
                roi_percent = ((final_value - initial_balance) / initial_balance) * 100

                # Better balanced score for long-term performance
                acceptable_max_drawdown = 0.20  # 20%
                score = (final_value / initial_balance) * (
                            1 - min(max_drawdown, acceptable_max_drawdown) / acceptable_max_drawdown)

                # Original score calculation (for reference)
                old_score = (final_value / initial_balance) * (1 - max_drawdown)

                # Track results
                trial_result = {
                    'trial_id': trial_id,
                    'timesteps': timesteps,
                    'dynamic_stop_loss': dynamic_stop_loss,
                    'reward_scaling': reward_scaling,
                    'lookback_window': lookback_window,
                    'final_value': final_value,
                    'roi_percent': roi_percent,
                    'max_drawdown': max_drawdown,
                    'original_score': old_score,
                    'score': score,
                    'runtime': time.time() - start_time
                }

                results.append(trial_result)

                # Update best if needed
                if final_value > best_final_value:
                    best_final_value = final_value
                    best_params = {
                        'timesteps': timesteps,
                        'dynamic_stop_loss': dynamic_stop_loss,
                        'reward_scaling': reward_scaling,
                        'lookback_window': lookback_window
                    }
                    # Copy best model
                    os.system(f"cp -r {model_path}/* ./models/best_model/")

                logger.info(f"Trial {trial_id} completed: "
                            f"final_value={final_value:.2f}, "
                            f"ROI={roi_percent:.2f}%, "
                            f"max_drawdown={max_drawdown:.4f}, "
                            f"runtime={trial_result['runtime']:.2f}s")

                # Print current best
                logger.info(f"Current best final value: {best_final_value:.4f} with params: {best_params}")

            else:
                logger.warning(f"Trial {trial_id} failed: No results returned")

        except Exception as e:
            logger.error(f"Trial {trial_id} failed with error: {str(e)}")
            results.append({
                'trial_id': trial_id,
                'timesteps': timesteps,
                'dynamic_stop_loss': dynamic_stop_loss,
                'reward_scaling': reward_scaling,
                'lookback_window': lookback_window,
                'final_value': -1,
                'roi_percent': -100,
                'max_drawdown': 1.0,
                'original_score': 0,
                'score': float('-inf'),
                'error': str(e),
                'runtime': time.time() - start_time
            })

        # Save intermediate results after each trial
        results_df = pd.DataFrame(results)
        results_df.to_csv(f"grid_search_results/{symbol}_{timeperiod}_results.csv", index=False)

    # Print final results
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('final_value', ascending=False)

    print("\n===== GRID SEARCH RESULTS =====")
    print(f"Total combinations tested: {len(results)}")
    print(f"Best final value: {best_final_value:.2f}")

    print("\nBest hyperparameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")

    print("\nTop 5 configurations by final portfolio value:")
    top5 = results_df.head(5)
    for _, row in top5.iterrows():
        print(f"  Final value: ${row['final_value']:.2f} (ROI: {row['roi_percent']:.2f}%), "
              f"params: timesteps={row['timesteps']}, "
              f"dynamic_stop_loss={row['dynamic_stop_loss']}, "
              f"reward_scaling={row['reward_scaling']}, "
              f"lookback_window={row['lookback_window']}")

    # Save final results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df.to_csv(f"grid_search_results/{symbol}_{timeperiod}_final_{timestamp}.csv", index=False)

    return best_params, results_df


def analyze_existing_results(results_file):
    """Analyze and print existing results sorted by final value"""

    try:
        # Load results file
        df = pd.read_csv(results_file)

        # Add ROI column if not exists
        if 'roi_percent' not in df.columns:
            df['roi_percent'] = ((df['final_value'] - 100000) / 100000) * 100

        # Sort by final value
        df_by_value = df.sort_values('final_value', ascending=False)

        # Print summary
        print("\n===== RESULTS ANALYSIS =====")
        print(f"Total configurations tested: {len(df)}")

        # Get best result by final value
        best = df_by_value.iloc[0]
        print(f"Best final value: ${best['final_value']:.2f} (ROI: {best['roi_percent']:.2f}%)")
        print("Best config parameters:")
        print(f"  timesteps: {best['timesteps']}")
        print(f"  dynamic_stop_loss: {best['dynamic_stop_loss']}")
        print(f"  reward_scaling: {best['reward_scaling']}")
        print(f"  lookback_window: {best['lookback_window']}")

        # Print comparison between final value and original score
        print("\nRelationship between final value and score:")
        print("The original score was calculated as: (final_value / initial_balance) * (1 - max_drawdown)")
        print("This means it rewards high returns while penalizing volatility")

        # Print top 10 by final value
        print("\nTop 10 configurations by final portfolio value:")
        for i, row in df_by_value.head(10).iterrows():
            print(f"  {row['trial_id']}: ${row['final_value']:.2f} (ROI: {row['roi_percent']:.2f}%), "
                  f"max_drawdown: {row['max_drawdown']:.4f}, "
                  f"params: timesteps={row['timesteps']}, "
                  f"dynamic_stop_loss={row['dynamic_stop_loss']}, "
                  f"reward_scaling={row['reward_scaling']}, "
                  f"lookback_window={row['lookback_window']}")

        return df_by_value

    except Exception as e:
        print(f"Error analyzing results file: {str(e)}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Grid search and analysis for trading agent')
    parser.add_argument('--symbol', type=str, default='SOL', help='Trading symbol')
    parser.add_argument('--timeperiod', type=int, default=5, help='Time period in minutes')
    parser.add_argument('--analyze', type=str, help='Analyze existing results file')

    args = parser.parse_args()

    if args.analyze:
        analyze_existing_results(args.analyze)
    else:
        best_params, results = run_grid_search(
            symbol=args.symbol,
            timeperiod=args.timeperiod
        )