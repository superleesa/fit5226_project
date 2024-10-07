# FIT5226 Project: Dynamic Grid World Agent with DQN

<img src="docs/sample_grid_world.png" alt="Grid World Image" width="400"/>

## Training / Fine-Tuning
```bash
python scripts/finetune.py  configs/config.yml  # train
python scripts/finetune.py configs/config.yml checkpoints/{episode_x}.pt  # fine-tune
mlflow ui  # monitor training (automatically tracked when you run above command)
```

## Evaluation and Visualization
```bash
python scripts/evaluate.py configs/eval_config.yml checkpoints/{episode_x}.pt
```

## Hyper-Parameter Tuning
```bash
python scripts/hp_tune.py study_name
optuna-dashboard sqlite:///tuning_result.db  # monitor hyperparameter tuning
```

## Installation
```bash
git clone https://github.com/superleesa/fit5226_project.git
pip install -e .
```

## Result
![Result](notebooks/a2/training_visualization.png)