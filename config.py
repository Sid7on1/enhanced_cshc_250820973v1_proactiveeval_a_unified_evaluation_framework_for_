import os
import logging
from typing import Dict, List
from dataclasses import dataclass

import numpy as np
import torch
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set seed for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Configuration class for the agent
@dataclass
class AgentConfig:
    """Configuration settings for the agent."""
    model_path: str = "models/proactive_agent.pt"
    vocab_size: int = 50000
    embedding_dim: int = 300
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 64
    num_epochs: int = 20
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __post_init__(self):
        """Validate and set default values for configuration settings."""
        self.validate()

    def validate(self):
        """Validate the configuration settings."""
        if not os.path.isfile(self.model_path):
            logger.error(f"Model file '{self.model_path}' does not exist.")
            exit(1)

        if self.vocab_size < 10000 or self.vocab_size > 100000:
            logger.warning("Vocab size is outside recommended range of 10000 to 100000.")

        if self.embedding_dim < 100 or self.embedding_dim > 500:
            logger.warning("Embedding dimension is outside recommended range of 100 to 500.")

        if self.hidden_dim < 64 or self.hidden_dim > 1024:
            logger.warning("Hidden dimension is outside recommended range of 64 to 1024.")

        if self.num_layers < 1 or self.num_layers > 5:
            logger.warning("Number of layers is outside recommended range of 1 to 5.")

        if not 0 <= self.dropout < 1:
            logger.error("Dropout rate must be between 0 and 1.")
            exit(1)

        if self.learning_rate <= 0:
            logger.error("Learning rate must be greater than 0.")
            exit(1)

        if self.batch_size < 8 or self.batch_size > 256:
            logger.warning("Batch size is outside recommended range of 8 to 256.")

        if self.num_epochs < 5:
            logger.warning("Number of epochs is below the recommended minimum of 5.")

# Configuration class for the environment
@dataclass
class EnvironmentConfig:
    """Configuration settings for the environment."""
    data_file: str = "data/proactive_dialogues.csv"
    max_dialogue_length: int = 10
    min_context_length: int = 2
    max_context_length: int = 4
    output_dir: str = "output/"
    evaluation_metrics: List[str] = ["accuracy", "f1_score", "roc_auc"]

    def __post_init__(self):
        """Validate and set default values for configuration settings."""
        self.validate()

    def validate(self):
        """Validate the configuration settings."""
        if not os.path.isfile(self.data_file):
            logger.error(f"Data file '{self.data_file}' does not exist.")
            exit(1)

        if self.max_dialogue_length < 5 or self.max_dialogue_length > 20:
            logger.warning("Max dialogue length is outside recommended range of 5 to 20.")

        if not (1 <= self.min_context_length <= self.max_context_length <= self.max_dialogue_length):
            logger.error("Invalid context length range.")
            exit(1)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        invalid_metrics = set(self.evaluation_metrics) - set(["accuracy", "f1_score", "roc_auc", "precision", "recall"])
        if invalid_metrics:
            logger.warning(f"Invalid evaluation metrics: {invalid_metrics}. Using default metrics.")
            self.evaluation_metrics = ["accuracy", "f1_score", "roc_auc"]

# Function to load configuration settings from files
def load_config(config_file: str) -> Dict:
    """
    Load configuration settings from a file.

    Parameters:
    - config_file (str): Path to the configuration file.

    Returns:
    - config (dict): Dictionary containing the configuration settings.
    """
    if not os.path.isfile(config_file):
        logger.error(f"Config file '{config_file}' does not exist.")
        exit(1)

    try:
        config = {}
        with open(config_file, "r") as file:
            for line in file:
                name, value = line.strip().split("=")
                config[name.strip()] = value.strip()
        return config
    except Exception as e:
        logger.error(f"Error loading config file: {e}")
        exit(1)

# Function to update configuration settings from a dictionary
def update_config(config: Dict, updates: Dict) -> Dict:
    """
    Update configuration settings with new values.

    Parameters:
    - config (dict): Original configuration settings.
    - updates (dict): Dictionary containing updated values.

    Returns:
    - updated_config (dict): Updated configuration settings.
    """
    updated_config = config.copy()
    updated_config.update(updates)
    return updated_config

# Function to initialize configuration settings
def initialize_config(config_file: str = None, updates: Dict = None) -> Dict:
    """
    Initialize configuration settings.

    Loads configuration from a file and applies updates if provided.

    Parameters:
    - config_file (str, optional): Path to the configuration file. Defaults to None.
    - updates (dict, optional): Dictionary containing updated values. Defaults to None.

    Returns:
    - config (dict): Final configuration settings.
    """
    if config_file:
        config = load_config(config_file)
    else:
        config = {
            "model_path": "models/proactive_agent.pt",
            "vocab_size": 50000,
            "embedding_dim": 300,
            "hidden_dim": 256,
            "num_layers": 2,
            "dropout": 0.2,
            "learning_rate": 0.001,
            "batch_size": 64,
            "num_epochs": 20,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "data_file": "data/proactive_dialogues.csv",
            "max_dialogue_length": 10,
            "min_context_length": 2,
            "max_context_length": 4,
            "output_dir": "output/",
            "evaluation_metrics": ["accuracy", "f1_score", "roc_auc"]
        }

    if updates:
        config = update_config(config, updates)

    return config

# Function to get configuration settings
def get_config(config: Dict, section: str) -> Dict:
    """
    Get configuration settings for a specific section.

    Parameters:
    - config (dict): Final configuration settings.
    - section (str): Name of the section to retrieve settings for.

    Returns:
    - settings (dict): Configuration settings for the specified section.
    """
    settings = {}
    for key, value in config.items():
        if key.startswith(section + "_"):
            settings[key[len(section) + 1:]] = value
    return settings

# Example usage
if __name__ == "__main__":
    # Initialize configuration settings
    config_file = "config.ini"
    updates = {
        "model_path": "new_model.pt",
        "learning_rate": 0.0005,
        "evaluation_metrics": ["f1_score", "precision"]
    }
    config = initialize_config(config_file, updates)

    # Get agent and environment configuration settings
    agent_config = get_config(config, "agent")
    agent = AgentConfig(**agent_config)
    logger.info(f"Agent Config: {agent_config}")

    environment_config = get_config(config, "environment")
    environment = EnvironmentConfig(**environment_config)
    logger.info(f"Environment Config: {environment_config}")