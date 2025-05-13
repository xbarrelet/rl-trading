Testing your environment:
Start by testing the environment without training a full model:
python test_env.py

Training your RL agent:
python main.py --mode train --symbol SOL --timeperiod 15 --timesteps 100000

Evaluating a trained model:
python main.py --mode evaluate --model_path ./models/final_model --render