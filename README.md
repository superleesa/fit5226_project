# FIT5226 Project: Dynamic Grid World Agent with DQN

![Grid World Image](docs/sample_grid_world.png)

Start fine-tuning with:
```bash
python scripts/finetune.py notebooks/a2/episode_250.pt
```

Monitor Training with:
```bash
mlflow ui
```

Evaluate it with:
```bash
python scripts/evaluate.py notebooks/a2/episode_250.pt
```

## Installation
```bash
pip install -e .
```

## Result
![Result](notebooks/a2/training_visualization.png)