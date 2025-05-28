import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import tkinter as tk
from tkinter import messagebox, ttk
import os
import threading
import time
from collections import deque


class TicTacToeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(9, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 9)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


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
        """Carrega modelo tratando incompatibilidades de arquitetura"""
        if not os.path.exists("tic_tac_toe_model.pth"):
            print("⚠️ Modelo ainda não treinado. Iniciando do zero.")
            return

        try:
            # Tenta carregar normalmente
            state_dict = torch.load("tic_tac_toe_model.pth")
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print("✅ Modelo carregado com sucesso.")
            
        except RuntimeError as e:
            print(f"⚠️ Incompatibilidade na arquitetura do modelo: {e}")
            
            # Verifica se é a arquitetura antiga (128->128->9)
            try:
                state_dict = torch.load("tic_tac_toe_model.pth")
                
                # Se tem fc4, é a arquitetura nova (incompatível)
                if 'fc4.weight' in state_dict:
                    print("🔄 Modelo tem arquitetura mais complexa. Fazendo backup e recomeçando...")
                    backup_name = f"tic_tac_toe_model_backup_{int(time.time())}.pth"
                    os.rename("tic_tac_toe_model.pth", backup_name)
                    print(f"📁 Backup salvo como: {backup_name}")
                    print("🆕 Iniciando com nova instância do modelo.")
                    
                # Se não tem fc4 mas tem tamanhos diferentes, também é incompatível
                elif state_dict['fc1.weight'].shape[0] != 128:
                    print("🔄 Modelo tem tamanhos de camadas diferentes. Fazendo backup...")
                    backup_name = f"tic_tac_toe_model_backup_{int(time.time())}.pth"
                    os.rename("tic_tac_toe_model.pth", backup_name)
                    print(f"📁 Backup salvo como: {backup_name}")
                    print("🆕 Iniciando com nova instância do modelo.")
                    
                else:
                    # Deve ser compatível, mas algo deu errado
                    print("❌ Erro inesperado ao carregar modelo. Iniciando do zero.")
                    
            except Exception as load_error:
                print(f"❌ Erro ao analisar modelo: {load_error}")
                print("🆕 Iniciando com modelo novo.")

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def choose_action(self, state, valid_actions, use_strategy=True, training=False):
        # Estratégia híbrida: regras + rede neural
        if use_strategy:
            critical_move = self.get_critical_move(state, valid_actions)
            if critical_move is not None:
                return critical_move
        
        # Exploração durante treinamento
        if training and random.random() < self.epsilon:
            return random.choice(valid_actions)
        
        # Usa rede neural
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor).squeeze()
        
        # Mascara ações inválidas
        masked_q_values = [-float('inf')] * 9
        for i in valid_actions:
            masked_q_values[i] = q_values[i].item()
        
        return np.argmax(masked_q_values)

    def get_critical_move(self, state, valid_actions):
        """Estratégias básicas de tic-tac-toe"""
        # 1. Vencer se possível
        for action in valid_actions:
            temp_state = state.copy()
            temp_state[action] = -1  # IA é -1
            if check_winner(temp_state) == -1:
                return action
        
        # 2. Bloquear vitória do oponente
        for action in valid_actions:
            temp_state = state.copy()
            temp_state[action] = 1  # Oponente é 1
            if check_winner(temp_state) == 1:
                return action
        
        # 3. Centro se disponível
        if 4 in valid_actions:
            return 4
        
        # 4. Cantos
        corners = [0, 2, 6, 8]
        available_corners = [c for c in corners if c in valid_actions]
        if available_corners:
            return random.choice(available_corners)
        
        return None

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

        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_model(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        loss = self.loss_fn(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        self.training_steps += 1
        if self.training_steps % self.update_target_freq == 0:
            self.update_target_model()


def get_valid_actions(board):
    return [i for i, x in enumerate(board) if x == 0]


def apply_move(board, move, player):
    new_board = board.copy()
    new_board[move] = player
    return new_board


def check_winner(board):
    winning_combos = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),  # Linhas
        (0, 3, 6), (1, 4, 7), (2, 5, 8),  # Colunas
        (0, 4, 8), (2, 4, 6)              # Diagonais
    ]
    for a, b, c in winning_combos:
        if board[a] == board[b] == board[c] and board[a] != 0:
            return board[a]
    return 0 if 0 in board else None


def get_reward(winner, player):
    """Sistema de recompensas simples"""
    if winner == player:
        return 10.0
    elif winner == -player:
        return -10.0
    elif winner is None:
        return 1.0
    else:
        return 0.0


class TicTacToeApp:
    def __init__(self):
        self.agent = DQNAgent()
        self.board = [0] * 9
        self.game_mode = "human_vs_ai"  # "human_vs_ai", "ai_vs_ai", "training"
        self.is_training = False
        self.training_stats = {'games': 0, 'ai1_wins': 0, 'ai2_wins': 0, 'draws': 0}
        self.human_stats = {'games': 0, 'human_wins': 0, 'ai_wins': 0, 'draws': 0}
        
        self.setup_gui()
        self.training_thread = None

    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("🎮 Tic Tac Toe AI - Versão Final")
        self.root.geometry("500x600")
        
        # Frame de controles
        control_frame = tk.Frame(self.root)
        control_frame.pack(pady=10)
        
        tk.Label(control_frame, text="Modo de Jogo:", font=("Arial", 12, "bold")).pack()
        
        # Botões de modo
        mode_frame = tk.Frame(control_frame)
        mode_frame.pack(pady=5)
        
        tk.Button(mode_frame, text="🧑 vs 🤖", command=lambda: self.set_mode("human_vs_ai"),
                 bg="lightblue", font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
        
        tk.Button(mode_frame, text="🤖 vs 🤖", command=lambda: self.set_mode("ai_vs_ai"),
                 bg="lightgreen", font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
        
        tk.Button(mode_frame, text="🎯 Treinar", command=lambda: self.set_mode("training"),
                 bg="orange", font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
        
        # Status
        self.status_label = tk.Label(self.root, text="Modo: Humano vs IA", font=("Arial", 14))
        self.status_label.pack(pady=5)
        
        # Grid do jogo
        self.setup_game_grid()
        
        # Estatísticas
        self.stats_label = tk.Label(self.root, text="", font=("Arial", 10))
        self.stats_label.pack(pady=5)
        
        # Controles de treinamento
        training_frame = tk.Frame(self.root)
        training_frame.pack(pady=10)
        
        self.training_label = tk.Label(training_frame, text="", font=("Arial", 10))
        self.training_label.pack()
        
        # Botão salvar modelo
        tk.Button(training_frame, text="💾 Salvar Modelo", command=self.save_model,
                 bg="yellow").pack(pady=5)

    def setup_game_grid(self):
        self.grid_frame = tk.Frame(self.root)
        self.grid_frame.pack(pady=20)
        
        self.buttons = []
        for i in range(9):
            btn = tk.Button(self.grid_frame, text="", font=("Arial", 20), 
                           width=5, height=2, command=lambda i=i: self.human_move(i))
            btn.grid(row=i//3, column=i%3, padx=2, pady=2)
            self.buttons.append(btn)

    def set_mode(self, mode):
        # Para treinamento se estiver rodando
        if self.is_training:
            self.is_training = False
            if self.training_thread and self.training_thread.is_alive():
                self.training_thread.join(timeout=1)
        
        self.game_mode = mode
        self.reset_game()
        
        if mode == "human_vs_ai":
            self.status_label.config(text="Modo: 🧑 Humano vs 🤖 IA | Sua vez!")
            for btn in self.buttons:
                btn.config(state="normal")
                
        elif mode == "ai_vs_ai":
            self.status_label.config(text="Modo: 🤖 IA vs 🤖 IA | Assistindo...")
            for btn in self.buttons:
                btn.config(state="disabled")
            self.root.after(1000, self.start_ai_vs_ai)
            
        elif mode == "training":
            self.status_label.config(text="Modo: 🎯 Treinamento | IA aprendendo...")
            for btn in self.buttons:
                btn.config(state="disabled")
            self.start_training()

    def human_move(self, index):
        if self.game_mode != "human_vs_ai" or self.board[index] != 0:
            return
        
        self.board[index] = 1  # Humano é X (1)
        self.update_display()
        
        winner = check_winner(self.board)
        if winner or 0 not in self.board:
            self.end_game(winner)
            return
        
        self.status_label.config(text="🤖 IA pensando...")
        self.root.after(500, self.ai_move)

    def ai_move(self, player=-1):
        valid = get_valid_actions(self.board)
        if not valid:
            return
        
        action = self.agent.choose_action(self.board, valid, use_strategy=True)
        self.board[action] = player
        self.update_display()
        
        winner = check_winner(self.board)
        if winner or 0 not in self.board:
            self.end_game(winner)
        else:
            if self.game_mode == "human_vs_ai":
                self.status_label.config(text="Sua vez! (Você é X)")

    def start_ai_vs_ai(self):
        if self.game_mode != "ai_vs_ai":
            return
        
        valid = get_valid_actions(self.board)
        if not valid:
            return
        
        # Alterna entre os jogadores (-1 e 1)
        current_count = 9 - len(valid)
        current_player = 1 if current_count % 2 == 0 else -1
        
        action = self.agent.choose_action(self.board, valid, use_strategy=True)
        self.board[action] = current_player
        self.update_display()
        
        winner = check_winner(self.board)
        if winner or 0 not in self.board:
            self.end_game(winner)
        else:
            player_name = "IA 1 (X)" if current_player == 1 else "IA 2 (O)"
            self.status_label.config(text=f"Jogando: {player_name}")
            self.root.after(1000, self.start_ai_vs_ai)

    def start_training(self):
        self.is_training = True
        self.training_thread = threading.Thread(target=self.training_loop)
        self.training_thread.daemon = True
        self.training_thread.start()

    def training_loop(self):
        games_per_update = 100
        
        while self.is_training:
            # Simula jogos para treinamento
            for _ in range(games_per_update):
                if not self.is_training:
                    break
                    
                result = self.simulate_training_game()
                self.training_stats['games'] += 1
                
                if result == 1:
                    self.training_stats['ai1_wins'] += 1
                elif result == -1:
                    self.training_stats['ai2_wins'] += 1
                else:
                    self.training_stats['draws'] += 1
            
            # Atualiza interface
            if self.is_training:
                self.root.after(0, self.update_training_display)
            
            time.sleep(0.1)  # Evita usar 100% CPU

    def simulate_training_game(self):
        """Simula um jogo para treinamento"""
        state = [0] * 9
        current_player = 1
        transitions = []
        
        while True:
            valid = get_valid_actions(state)
            if not valid:
                break
            
            old_state = state.copy()
            
            # IA joga com exploração
            action = self.agent.choose_action(state, valid, use_strategy=True, training=True)
            state = apply_move(state, action, current_player)
            
            winner = check_winner(state)
            done = winner is not None or 0 not in state
            
            # Armazena transição
            reward = get_reward(winner, current_player)
            transitions.append((old_state, action, reward, state, done))
            
            if done:
                break
                
            current_player *= -1
        
        # Armazena todas as transições
        for transition in transitions:
            self.agent.store_transition(*transition)
        
        # Treina a rede
        for _ in range(3):
            self.agent.train_step()
        
        return winner

    def update_display(self):
        for i, val in enumerate(self.board):
            text = "X" if val == 1 else "O" if val == -1 else ""
            color = "blue" if val == 1 else "red" if val == -1 else "black"
            self.buttons[i].config(text=text, fg=color)

    def update_training_display(self):
        stats = self.training_stats
        text = f"🎯 Treinamento: {stats['games']} jogos | IA1: {stats['ai1_wins']} | IA2: {stats['ai2_wins']} | Empates: {stats['draws']}"
        self.training_label.config(text=text)

    def end_game(self, winner):
        # Atualiza estatísticas
        if self.game_mode == "human_vs_ai":
            self.human_stats['games'] += 1
            if winner == 1:
                self.human_stats['human_wins'] += 1
                self.status_label.config(text="🎉 Você venceu!")
            elif winner == -1:
                self.human_stats['ai_wins'] += 1
                self.status_label.config(text="🤖 IA venceu!")
            else:
                self.human_stats['draws'] += 1
                self.status_label.config(text="🤝 Empate!")
            
            # Mostra estatísticas
            stats = self.human_stats
            total = stats['games']
            human_pct = (stats['human_wins'] / total) * 100 if total > 0 else 0
            ai_pct = (stats['ai_wins'] / total) * 100 if total > 0 else 0
            draw_pct = (stats['draws'] / total) * 100 if total > 0 else 0
            
            stats_text = f"Jogos: {total} | Você: {stats['human_wins']} ({human_pct:.1f}%) | IA: {stats['ai_wins']} ({ai_pct:.1f}%) | Empates: {stats['draws']} ({draw_pct:.1f}%)"
            self.stats_label.config(text=stats_text)
            
        elif self.game_mode == "ai_vs_ai":
            if winner == 1:
                self.status_label.config(text="🏆 IA 1 (X) venceu!")
            elif winner == -1:
                self.status_label.config(text="🏆 IA 2 (O) venceu!")
            else:
                self.status_label.config(text="🤝 Empate entre IAs!")
        
        # Desabilita botões temporariamente
        for btn in self.buttons:
            btn.config(state="disabled")
        
        # Reset automático
        if self.game_mode in ["human_vs_ai", "ai_vs_ai"]:
            self.root.after(2000, self.reset_game)

    def reset_game(self):
        self.board = [0] * 9
        self.update_display()
        
        if self.game_mode == "human_vs_ai":
            for btn in self.buttons:
                btn.config(state="normal")
            self.status_label.config(text="Sua vez! (Você é X)")
        elif self.game_mode == "ai_vs_ai":
            self.root.after(1000, self.start_ai_vs_ai)

    def save_model(self):
        torch.save(self.agent.model.state_dict(), "tic_tac_toe_model.pth")
        messagebox.showinfo("Sucesso", "💾 Modelo salvo com sucesso!")

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

    def on_closing(self):
        self.is_training = False
        if self.training_thread and self.training_thread.is_alive():
            self.training_thread.join(timeout=1)
        self.root.destroy()


if __name__ == "__main__":
    print("🚀 Iniciando Tic Tac Toe AI...")
    app = TicTacToeApp()
    app.run()