import optuna
import random
import numpy as np


def DQN_Hyperparameter_Tune(trial: optuna.Trial) -> float:
    # Hyperparameters to optimize
    alpha = trial.suggest_uniform('alpha', 0.995, 0.999)
    discount_rate = trial.suggest_uniform('discount_rate', 0.95, 0.975)
    epsilon_decay = trial.suggest_uniform('epsilon_decay', 0.975, 0.995)
    replay_memory_size = trial.suggest_int('replay_memory_size', 1000, 5000)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128, 256])
    update_target_steps = trial.suggest_int('update_target_steps', 500, 1000)
    time_penalty = trial.suggest_int('time_penalty', -10, -1)
    item_state_reward = trial.suggest_int('item_state_reward', 100, 400)
    goal_state_reward = trial.suggest_int('goal_state_reward', 300, 600)

    tune_env = Assignment2Environment(
      n=4,  # Grid size
      time_penalty=time_penalty,
      item_state_reward=item_state_reward,
      goal_state_reward=goal_state_reward,
      direction_reward_multiplier=1,
      with_animation=False 
    ) 
    tune_agent = DQNAgent(
      statespace_size=11,
      action_space_size=len(Action),
      alpha=alpha,
      discount_rate=discount_rate,
      epsilon=1.0, 
      epsilon_decay=epsilon_decay,
      epsilon_min=0.1,
      replay_memory_size=replay_memory_size,
      batch_size=batch_size,
      update_target_steps=update_target_steps
    )

    num_episodes = 200
    tune_trainer = Trainer(tune_agent, tune_env)

    current_state = tune_env.environment.get_state()  # Get state from current sub-environment
    done = False
    total_reward = 0  # Track total reward for the episode
    step_count = 0  # Initialize step counter

    while not done:  # Truncate episode after 40 steps
        # Convert the current state to a numpy array for input to the neural network
        state_array = tune_trainer.state_to_array(current_state)

        # Retrieve available actions from the current sub-environment
        available_actions = tune_trainer.environment.get_available_actions(current_state)

        # Select an action using the agent's ε-greedy policy
        action = tune_trainer.agent.select_action(state_array, available_actions)

        # Execute the action in the current sub-environment, receive reward and next state
        reward, next_state = tune_trainer.environment.step(action)

        # Add the reward to the total reward for this episode
        total_reward += reward

        # Convert the next state to a numpy array
        next_state_array = tune_trainer.state_to_array(next_state)

        # Check if the next state is a goal state
        done = tune_trainer.environment.is_goal_state(next_state)

        # Store experience in the agent's replay memory
        tune_trainer.agent.remember((state_array, action.value, reward, next_state_array, done))

        # Learn from experiences using experience replay
        tune_trainer.agent.replay()

        # Move to the next state
        current_state = next_state

        # Increment the step counter
        step_count += 1

        # Update epsilon to decrease exploration over time
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

        # Store total reward of the episode
        tune_trainer.episode_rewards.append(total_reward)
    
        if trial.should_prune():
          raise optuna.exceptions.TrialPruned()
    
    return total_reward


def main():
    # Create an Optuna study
    study = optuna.create_study(
        direction='maximize',  # We want to maximize the average reward
    )
    
    # Optimize the objective function
    study.optimize(DQN_Hyperparameter_Tune)
    
    # Print the best hyperparameters and the best value
    print("Best Hyperparameters: ", study.best_params)
    print("Best Value: ", study.best_value)

if __name__ == "__main__":
    main()
