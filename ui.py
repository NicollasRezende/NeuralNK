from agent import DQNAgent
from game import get_valid_actions, apply_move, check_winner, get_reward
import tkinter as tk
import threading
import torch
from tkinter import messagebox

class TicTacToeApp:
    def __init__(self):
        self.agent = DQNAgent()
        self.board = [0] * 9
        self.game_mode = "human_vs_ai"
        self.is_training = False
        self.training_thread = None
        self.training_stats = {'games': 0, 'ai1_wins': 0, 'ai2_wins': 0, 'draws': 0}
        self.human_stats = {'games': 0, 'human_wins': 0, 'ai_wins': 0, 'draws': 0}
        self.setup_gui()

    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("Tic Tac Toe AI")
        self.root.geometry("500x600")

        mode_frame = tk.Frame(self.root)
        mode_frame.pack(pady=10)

        tk.Button(mode_frame, text="🧑 vs 🤖", command=lambda: self.set_mode("human_vs_ai")).pack(side=tk.LEFT)
        tk.Button(mode_frame, text="🤖 vs 🤖", command=lambda: self.set_mode("ai_vs_ai")).pack(side=tk.LEFT)
        tk.Button(mode_frame, text="🎯 Treinar", command=lambda: self.set_mode("training")).pack(side=tk.LEFT)

        self.status_label = tk.Label(self.root, text="Modo: Humano vs IA")
        self.status_label.pack(pady=5)

        self.grid_frame = tk.Frame(self.root)
        self.grid_frame.pack(pady=20)
        self.buttons = []
        for i in range(9):
            btn = tk.Button(self.grid_frame, text="", font=("Arial", 20), width=5, height=2, command=lambda i=i: self.human_move(i))
            btn.grid(row=i//3, column=i%3)
            self.buttons.append(btn)

        self.stats_label = tk.Label(self.root, text="")
        self.stats_label.pack(pady=5)

        training_frame = tk.Frame(self.root)
        training_frame.pack()
        self.training_label = tk.Label(training_frame, text="")
        self.training_label.pack()
        tk.Button(training_frame, text="💾 Salvar Modelo", command=self.save_model).pack(pady=5)

    def set_mode(self, mode):
        self.is_training = False
        if self.training_thread and self.training_thread.is_alive():
            self.training_thread.join(timeout=1)
        self.game_mode = mode
        self.reset_game()
        if mode == "ai_vs_ai":
            self.root.after(1000, self.start_ai_vs_ai)
        elif mode == "training":
            self.start_training()

    def human_move(self, index):
        if self.board[index] != 0 or self.game_mode != "human_vs_ai":
            return
        self.board[index] = 1
        self.update_display()
        winner = check_winner(self.board)
        if winner or 0 not in self.board:
            self.end_game(winner)
        else:
            self.root.after(500, self.ai_move)

    def ai_move(self, player=-1):
        valid = get_valid_actions(self.board)
        action = self.agent.choose_action(self.board, valid)
        self.board[action] = player
        self.update_display()
        winner = check_winner(self.board)
        if winner or 0 not in self.board:
            self.end_game(winner)

    def start_ai_vs_ai(self):
        if self.game_mode != "ai_vs_ai":
            return
        valid = get_valid_actions(self.board)
        current_player = 1 if (9 - len(valid)) % 2 == 0 else -1
        action = self.agent.choose_action(self.board, valid)
        self.board[action] = current_player
        self.update_display()
        winner = check_winner(self.board)
        if winner or 0 not in self.board:
            self.end_game(winner)
        else:
            self.root.after(1000, self.start_ai_vs_ai)

    def start_training(self):
        self.is_training = True
        self.training_thread = threading.Thread(target=self.training_loop)
        self.training_thread.daemon = True
        self.training_thread.start()

    def training_loop(self):
        while self.is_training:
            result = self.simulate_training_game()
            self.training_stats['games'] += 1
            if result == 1:
                self.training_stats['ai1_wins'] += 1
            elif result == -1:
                self.training_stats['ai2_wins'] += 1
            else:
                self.training_stats['draws'] += 1
            self.root.after(0, self.update_training_display)

    def simulate_training_game(self):
        state = [0] * 9
        current_player = 1
        transitions = []
        while True:
            valid = get_valid_actions(state)
            old_state = state.copy()
            action = self.agent.choose_action(state, valid, training=True)
            state = apply_move(state, action, current_player)
            winner = check_winner(state)
            done = winner is not None or 0 not in state
            reward = get_reward(winner, current_player)
            transitions.append((old_state, action, reward, state, done))
            if done:
                break
            current_player *= -1
        for trans in transitions:
            self.agent.store_transition(*trans)
        for _ in range(3):
            self.agent.train_step()
        return winner

    def update_display(self):
        for i, val in enumerate(self.board):
            text = "X" if val == 1 else "O" if val == -1 else ""
            self.buttons[i].config(text=text)

    def update_training_display(self):
        s = self.training_stats
        self.training_label.config(text=f"Jogos: {s['games']} | IA1: {s['ai1_wins']} | IA2: {s['ai2_wins']} | Empates: {s['draws']}")

    def end_game(self, winner):
        for btn in self.buttons:
            btn.config(state="disabled")
        self.root.after(2000, self.reset_game)

    def reset_game(self):
        self.board = [0] * 9
        self.update_display()
        for btn in self.buttons:
            btn.config(state="normal")
        if self.game_mode == "ai_vs_ai":
            self.root.after(1000, self.start_ai_vs_ai)

    def save_model(self):
        torch.save(self.agent.model.state_dict(), "tic_tac_toe_model.pth")
        messagebox.showinfo("Sucesso", "Modelo salvo com sucesso!")

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

    def on_closing(self):
        self.is_training = False
        if self.training_thread and self.training_thread.is_alive():
            self.training_thread.join(timeout=1)
        self.root.destroy()