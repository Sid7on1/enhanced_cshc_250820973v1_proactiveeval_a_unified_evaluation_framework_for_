import logging
import os
import random
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants and configuration
CONFIG = {
    'model_path': 'model.pth',
    'batch_size': 32,
    'epochs': 10,
    'learning_rate': 0.001,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

class ProactiveEvalDataset(Dataset):
    def __init__(self, data: pd.DataFrame, target_planning: bool = True):
        self.data = data
        self.target_planning = target_planning

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        item = self.data.iloc[idx]
        if self.target_planning:
            # Target planning
            input_seq = item['input_seq']
            target_seq = item['target_seq']
            return {
                'input_seq': input_seq,
                'target_seq': target_seq,
                'label': 1,
            }
        else:
            # Dialogue guidance
            input_seq = item['input_seq']
            output_seq = item['output_seq']
            return {
                'input_seq': input_seq,
                'output_seq': output_seq,
                'label': 1,
            }

class ProactiveEvalModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(ProactiveEvalModel, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, output_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        # Encoder
        h0 = torch.zeros(1, input_seq.size(0), self.hidden_dim).to(self.device)
        c0 = torch.zeros(1, input_seq.size(0), self.hidden_dim).to(self.device)
        output, _ = self.encoder(input_seq, (h0, c0))
        # Decoder
        h0 = torch.zeros(1, output.size(0), self.hidden_dim).to(self.device)
        c0 = torch.zeros(1, output.size(0), self.hidden_dim).to(self.device)
        output, _ = self.decoder(output, (h0, c0))
        # Output
        output = self.fc(output[:, -1, :])
        return output

class ProactiveEvalAgent:
    def __init__(self, config: Dict):
        self.config = config
        self.model = ProactiveEvalModel(input_dim=128, hidden_dim=256, output_dim=128)
        self.device = torch.device(self.config['device'])
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        self.criterion = nn.CrossEntropyLoss()

    def train(self, train_data: pd.DataFrame, val_data: pd.DataFrame):
        train_dataset = ProactiveEvalDataset(train_data, target_planning=True)
        val_dataset = ProactiveEvalDataset(val_data, target_planning=True)
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'], shuffle=False)
        for epoch in range(self.config['epochs']):
            self.model.train()
            total_loss = 0
            for batch in train_loader:
                input_seq = batch['input_seq'].to(self.device)
                target_seq = batch['target_seq'].to(self.device)
                label = batch['label'].to(self.device)
                self.optimizer.zero_grad()
                output = self.model(input_seq)
                loss = self.criterion(output, label)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    input_seq = batch['input_seq'].to(self.device)
                    target_seq = batch['target_seq'].to(self.device)
                    label = batch['label'].to(self.device)
                    output = self.model(input_seq)
                    loss = self.criterion(output, label)
                    val_loss += loss.item()
            logger.info(f'Epoch {epoch+1}, Train Loss: {total_loss / len(train_loader)}, Val Loss: {val_loss / len(val_loader)}')

    def evaluate(self, test_data: pd.DataFrame):
        test_dataset = ProactiveEvalDataset(test_data, target_planning=True)
        test_loader = DataLoader(test_dataset, batch_size=self.config['batch_size'], shuffle=False)
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                input_seq = batch['input_seq'].to(self.device)
                target_seq = batch['target_seq'].to(self.device)
                label = batch['label'].to(self.device)
                output = self.model(input_seq)
                loss = self.criterion(output, label)
                total_loss += loss.item()
        logger.info(f'Test Loss: {total_loss / len(test_loader)}')

def main():
    # Load data
    train_data = pd.read_csv('train.csv')
    val_data = pd.read_csv('val.csv')
    test_data = pd.read_csv('test.csv')

    # Create agent
    agent = ProactiveEvalAgent(CONFIG)

    # Train agent
    agent.train(train_data, val_data)

    # Evaluate agent
    agent.evaluate(test_data)

    # Save model
    torch.save(agent.model.state_dict(), CONFIG['model_path'])

if __name__ == '__main__':
    main()