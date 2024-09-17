import matplotlib.pyplot as plt
import os
from fit5226_project.agent import DQNAgent

class Plotter:
    def __init__(self, agent: DQNAgent, refresh_interval: int = 5000, save_dir: str = "./plots"):
        """
        Initialize the Plotter with the DQNAgent instance and refresh interval.
        
        :param agent: Instance of the DQNAgent to fetch data from.
        :param refresh_interval: Time interval (in milliseconds) to refresh the plot.
        :param save_dir: Directory where plots will be saved.
        """
        self.agent = agent
        self.refresh_interval = refresh_interval
        self.save_dir = save_dir

        # Create the save directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)

        # Set up the plot
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 10))
        self.fig.suptitle('DQN Agent Metrics Tracking')

    def update_plot(self):
        """
        Update the plot with new data.
        """
        # Fetch the latest metrics from the agent
        data = self.agent.logged_data
        print(data)
        
        # Clear the previous plots
        self.ax1.clear()
        self.ax2.clear()

        # Plot average predicted Q-values and target Q-values
        if data['avg_predicted_qval'] and data['avg_target_qval']:
            self.ax1.plot(data['steps'], data['avg_predicted_qval'], label='Avg Predicted Q-Value', color='blue')
            self.ax1.plot(data['steps'], data['avg_target_qval'], label='Avg Target Q-Value', color='green')
            self.ax1.set_xlabel('Step')
            self.ax1.set_ylabel('Q-Value')
            self.ax1.set_title('Average Q-Values over Steps')
            self.ax1.legend()

        # Plot loss
        if data['loss']:
            self.ax2.plot(data['steps'], data['loss'], label='Loss', color='red')
            self.ax2.set_xlabel('Step')
            self.ax2.set_ylabel('Loss')
            self.ax2.set_title('Training Loss over Steps')
            self.ax2.legend()

        # Refresh the plot
        self.fig.canvas.draw()

        # Save the plot to a file
        self.save_plot()

    def save_plot(self):
        """
        Save the current plot to a file in the specified directory.
        """
        filename = os.path.join(self.save_dir, f"agent_metrics.png")
        self.fig.savefig(filename)
        print(f"Plot saved to {filename}")

    def start(self):
        """
        Start the plot updates.
        """
        self.update_plot()  # Update and save the plot once
        plt.show()
