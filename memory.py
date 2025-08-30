import logging
import numpy as np
import pandas as pd
import torch
from typing import List, Tuple, Dict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExperienceReplayMemory:
    """
    Experience replay memory class.

    This class is responsible for storing and retrieving experiences from the agent's interactions with the environment.
    """

    def __init__(self, capacity: int, batch_size: int, gamma: float, epsilon: float):
        """
        Initialize the experience replay memory.

        Args:
        - capacity (int): The maximum number of experiences to store in the memory.
        - batch_size (int): The number of experiences to retrieve at a time.
        - gamma (float): The discount factor for the reward.
        - epsilon (float): The exploration rate.
        """
        self.capacity = capacity
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.experiences = []
        self.index = 0

    def add_experience(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """
        Add a new experience to the memory.

        Args:
        - state (np.ndarray): The current state of the environment.
        - action (int): The action taken by the agent.
        - reward (float): The reward received by the agent.
        - next_state (np.ndarray): The next state of the environment.
        - done (bool): Whether the episode is done.
        """
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        }
        if len(self.experiences) < self.capacity:
            self.experiences.append(experience)
        else:
            self.experiences[self.index] = experience
            self.index = (self.index + 1) % self.capacity

    def sample_experiences(self) -> List[Dict]:
        """
        Sample a batch of experiences from the memory.

        Returns:
        - A list of experiences.
        """
        indices = np.random.choice(len(self.experiences), self.batch_size, replace=False)
        return [self.experiences[i] for i in indices]

    def __len__(self):
        return len(self.experiences)


class ProactiveEvalMemory:
    """
    Proactive evaluation memory class.

    This class is responsible for storing and retrieving proactive evaluation metrics.
    """

    def __init__(self, capacity: int, batch_size: int, gamma: float, epsilon: float):
        """
        Initialize the proactive evaluation memory.

        Args:
        - capacity (int): The maximum number of experiences to store in the memory.
        - batch_size (int): The number of experiences to retrieve at a time.
        - gamma (float): The discount factor for the reward.
        - epsilon (float): The exploration rate.
        """
        self.capacity = capacity
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.metrics = []
        self.index = 0

    def add_metric(self, metric: float):
        """
        Add a new metric to the memory.

        Args:
        - metric (float): The proactive evaluation metric.
        """
        if len(self.metrics) < self.capacity:
            self.metrics.append(metric)
        else:
            self.metrics[self.index] = metric
            self.index = (self.index + 1) % self.capacity

    def sample_metrics(self) -> List[float]:
        """
        Sample a batch of metrics from the memory.

        Returns:
        - A list of metrics.
        """
        indices = np.random.choice(len(self.metrics), self.batch_size, replace=False)
        return [self.metrics[i] for i in indices]

    def __len__(self):
        return len(self.metrics)


class VelocityThresholdMemory:
    """
    Velocity threshold memory class.

    This class is responsible for storing and retrieving velocity threshold metrics.
    """

    def __init__(self, capacity: int, batch_size: int, gamma: float, epsilon: float):
        """
        Initialize the velocity threshold memory.

        Args:
        - capacity (int): The maximum number of experiences to store in the memory.
        - batch_size (int): The number of experiences to retrieve at a time.
        - gamma (float): The discount factor for the reward.
        - epsilon (float): The exploration rate.
        """
        self.capacity = capacity
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.metrics = []
        self.index = 0

    def add_metric(self, metric: float):
        """
        Add a new metric to the memory.

        Args:
        - metric (float): The velocity threshold metric.
        """
        if len(self.metrics) < self.capacity:
            self.metrics.append(metric)
        else:
            self.metrics[self.index] = metric
            self.index = (self.index + 1) % self.capacity

    def sample_metrics(self) -> List[float]:
        """
        Sample a batch of metrics from the memory.

        Returns:
        - A list of metrics.
        """
        indices = np.random.choice(len(self.metrics), self.batch_size, replace=False)
        return [self.metrics[i] for i in indices]

    def __len__(self):
        return len(self.metrics)


class FlowTheoryMemory:
    """
    Flow theory memory class.

    This class is responsible for storing and retrieving flow theory metrics.
    """

    def __init__(self, capacity: int, batch_size: int, gamma: float, epsilon: float):
        """
        Initialize the flow theory memory.

        Args:
        - capacity (int): The maximum number of experiences to store in the memory.
        - batch_size (int): The number of experiences to retrieve at a time.
        - gamma (float): The discount factor for the reward.
        - epsilon (float): The exploration rate.
        """
        self.capacity = capacity
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.metrics = []
        self.index = 0

    def add_metric(self, metric: float):
        """
        Add a new metric to the memory.

        Args:
        - metric (float): The flow theory metric.
        """
        if len(self.metrics) < self.capacity:
            self.metrics.append(metric)
        else:
            self.metrics[self.index] = metric
            self.index = (self.index + 1) % self.capacity

    def sample_metrics(self) -> List[float]:
        """
        Sample a batch of metrics from the memory.

        Returns:
        - A list of metrics.
        """
        indices = np.random.choice(len(self.metrics), self.batch_size, replace=False)
        return [self.metrics[i] for i in indices]

    def __len__(self):
        return len(self.metrics)


class MemoryManager:
    """
    Memory manager class.

    This class is responsible for managing the different types of memories.
    """

    def __init__(self, capacity: int, batch_size: int, gamma: float, epsilon: float):
        """
        Initialize the memory manager.

        Args:
        - capacity (int): The maximum number of experiences to store in the memory.
        - batch_size (int): The number of experiences to retrieve at a time.
        - gamma (float): The discount factor for the reward.
        - epsilon (float): The exploration rate.
        """
        self.capacity = capacity
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.experience_replay_memory = ExperienceReplayMemory(capacity, batch_size, gamma, epsilon)
        self.proactive_eval_memory = ProactiveEvalMemory(capacity, batch_size, gamma, epsilon)
        self.velocity_threshold_memory = VelocityThresholdMemory(capacity, batch_size, gamma, epsilon)
        self.flow_theory_memory = FlowTheoryMemory(capacity, batch_size, gamma, epsilon)

    def add_experience(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """
        Add a new experience to the experience replay memory.

        Args:
        - state (np.ndarray): The current state of the environment.
        - action (int): The action taken by the agent.
        - reward (float): The reward received by the agent.
        - next_state (np.ndarray): The next state of the environment.
        - done (bool): Whether the episode is done.
        """
        self.experience_replay_memory.add_experience(state, action, reward, next_state, done)

    def add_proactive_eval_metric(self, metric: float):
        """
        Add a new proactive evaluation metric to the proactive evaluation memory.

        Args:
        - metric (float): The proactive evaluation metric.
        """
        self.proactive_eval_memory.add_metric(metric)

    def add_velocity_threshold_metric(self, metric: float):
        """
        Add a new velocity threshold metric to the velocity threshold memory.

        Args:
        - metric (float): The velocity threshold metric.
        """
        self.velocity_threshold_memory.add_metric(metric)

    def add_flow_theory_metric(self, metric: float):
        """
        Add a new flow theory metric to the flow theory memory.

        Args:
        - metric (float): The flow theory metric.
        """
        self.flow_theory_memory.add_metric(metric)

    def sample_experiences(self) -> List[Dict]:
        """
        Sample a batch of experiences from the experience replay memory.

        Returns:
        - A list of experiences.
        """
        return self.experience_replay_memory.sample_experiences()

    def sample_proactive_eval_metrics(self) -> List[float]:
        """
        Sample a batch of proactive evaluation metrics from the proactive evaluation memory.

        Returns:
        - A list of metrics.
        """
        return self.proactive_eval_memory.sample_metrics()

    def sample_velocity_threshold_metrics(self) -> List[float]:
        """
        Sample a batch of velocity threshold metrics from the velocity threshold memory.

        Returns:
        - A list of metrics.
        """
        return self.velocity_threshold_memory.sample_metrics()

    def sample_flow_theory_metrics(self) -> List[float]:
        """
        Sample a batch of flow theory metrics from the flow theory memory.

        Returns:
        - A list of metrics.
        """
        return self.flow_theory_memory.sample_metrics()


def main():
    # Create a memory manager
    memory_manager = MemoryManager(capacity=1000, batch_size=32, gamma=0.99, epsilon=0.1)

    # Add some experiences to the experience replay memory
    for i in range(100):
        state = np.random.rand(10)
        action = np.random.randint(0, 10)
        reward = np.random.rand()
        next_state = np.random.rand(10)
        done = np.random.choice([True, False])
        memory_manager.add_experience(state, action, reward, next_state, done)

    # Add some proactive evaluation metrics to the proactive evaluation memory
    for i in range(100):
        metric = np.random.rand()
        memory_manager.add_proactive_eval_metric(metric)

    # Add some velocity threshold metrics to the velocity threshold memory
    for i in range(100):
        metric = np.random.rand()
        memory_manager.add_velocity_threshold_metric(metric)

    # Add some flow theory metrics to the flow theory memory
    for i in range(100):
        metric = np.random.rand()
        memory_manager.add_flow_theory_metric(metric)

    # Sample some experiences from the experience replay memory
    experiences = memory_manager.sample_experiences()
    print("Sampled experiences:")
    for experience in experiences:
        print(experience)

    # Sample some proactive evaluation metrics from the proactive evaluation memory
    metrics = memory_manager.sample_proactive_eval_metrics()
    print("Sampled proactive evaluation metrics:")
    print(metrics)

    # Sample some velocity threshold metrics from the velocity threshold memory
    metrics = memory_manager.sample_velocity_threshold_metrics()
    print("Sampled velocity threshold metrics:")
    print(metrics)

    # Sample some flow theory metrics from the flow theory memory
    metrics = memory_manager.sample_flow_theory_metrics()
    print("Sampled flow theory metrics:")
    print(metrics)


if __name__ == "__main__":
    main()