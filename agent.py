import os
import time
import torch
import random
import numpy as np
from collections import deque
from model import TicTacToeNet
from game import check_winner
import torch.nn as nn
import torch.optim as optim

class DQNAgent:
    def __init__(self, lr=0.001, gamma=0.95, epsilon=0.1):
        self.model = TicTacToeNet()
        self.target_model = TicTacToeNet()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.gamma = gamma
        self.epsilon = epsilon
        self.memory = deque(maxlen=30000)
        self.batch_size = 64
        self.update_target_freq = 50
        self.training_steps = 0

        self.load_model()
        self.update_target_model()

    def load_model(self):
        if not os.path.exists("tic_tac_toe_model.pth"):
            print("Modelo não encontrado. Iniciando do zero.")
            return
        try:
            state_dict = torch.load("tic_tac_toe_model.pth")
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print("Modelo carregado com sucesso.")
        except Exception as e:
            print("Erro ao carregar modelo:", e)
            backup = f"tic_tac_toe_model_backup_{int(time.time())}.pth"
            os.rename("tic_tac_toe_model.pth", backup)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def choose_action(self, state, valid_actions, use_strategy=True, training=False):
        if use_strategy:
            critical = self.get_critical_move(state, valid_actions)
            if critical is not None:
                return critical
        if training and random.random() < self.epsilon:
            return random.choice(valid_actions)
        with torch.no_grad():
            q_values = self.model(torch.FloatTensor(state).unsqueeze(0)).squeeze()
        masked_q = [-float('inf')] * 9
        for i in valid_actions:
            masked_q[i] = q_values[i].item()
        return np.argmax(masked_q)

    def get_critical_move(self, state, valid_actions):
        for action in valid_actions:
            temp = state.copy()
            temp[action] = -1
            if check_winner(temp) == -1:
                return action
        for action in valid_actions:
            temp = state.copy()
            temp[action] = 1
            if check_winner(temp) == 1:
                return action
        if 4 in valid_actions:
            return 4
        corners = [c for c in [0, 2, 6, 8] if c in valid_actions]
        return random.choice(corners) if corners else None

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones)

        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        next_q = self.target_model(next_states).max(1)[0].detach()
        target_q = rewards + self.gamma * next_q * ~dones

        loss = self.loss_fn(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        self.training_steps += 1
        if self.training_steps % self.update_target_freq == 0:
            self.update_target_model()