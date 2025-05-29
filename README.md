
# 🤖 Jogo da Velha com Inteligência Artificial (DQN)

Este projeto implementa um agente de IA treinado com **Deep Q-Learning (DQN)** para jogar **Tic-Tac-Toe (Jogo da Velha)**. A interface gráfica é feita com **Tkinter**, e o modelo de rede neural é desenvolvido com **PyTorch**.

---

## 📂 Estrutura do Projeto

```

📁 projeto/
├── main.py             # Ponto de entrada que inicia a aplicação
├── model.py            # Arquitetura da rede neural (TicTacToeNet)
├── agent.py            # Implementação do agente DQN
├── game.py             # Regras do jogo, verificação de vitória e sistema de recompensa
├── ui.py               # Interface gráfica completa com modos de jogo e treinamento
├── tic_tac_toe_model.pth # Arquivo de modelo salvo
├── README.md           # Documentação do projeto

````

---

## 🧠 Componentes

### 🔸 `model.py`: Rede Neural (TicTacToeNet)

Modelo com 3 camadas lineares e funções de ativação ReLU, responsável por processar o estado do tabuleiro e gerar Q-values para cada ação.

```python
Input:  [9]  # Estado do tabuleiro
Output: [9]  # Q-values para cada posição
````

---

### 🔸 `agent.py`: Agente DQN

Responsável por:

* Tomar decisões usando política epsilon-greedy
* Treinar com minibatches de um buffer de replay
* Manter uma rede de destino (target network) para maior estabilidade
* Aplicar heurísticas como movimentos críticos

---

### 🔸 `game.py`: Lógica do Jogo

Contém funções para:

* Validar ações possíveis
* Aplicar jogadas
* Verificar vencedor
* Calcular recompensa com base no resultado da partida

---

### 🔸 `ui.py`: Interface Gráfica

Interface interativa em Tkinter com suporte para:

* 🧑 vs 🤖: Humano contra IA
* 🤖 vs 🤖: Duelo entre IAs
* 🎯 Treinar: Treinamento contínuo da IA contra ela mesma
* Exibição de estatísticas e botão de salvar modelo

---

## 🚀 Como Executar

### 1. Instalar dependências

```bash
pip install torch numpy
```

### 2. Executar o jogo

```bash
python main.py
```

> Observação: `tkinter` já está incluso na maioria das instalações do Python. Em distribuições Linux minimalistas, instale via pacote `python3-tk`.

---

## 🎮 Modos de Jogo

| Modo       | Descrição                                 |
| ---------- | ----------------------------------------- |
| 🧑 vs 🤖   | Humano joga contra IA                     |
| 🤖 vs 🤖   | IA joga contra si mesma                   |
| 🎯 Treinar | IA joga partidas aleatórias para aprender |

---

## 💾 Salvamento de Modelo

Você pode salvar o progresso da IA clicando em **💾 Salvar Modelo**. O arquivo `tic_tac_toe_model.pth` será criado ou sobrescrito.

---

## 📊 Estatísticas

Exibição em tempo real de:

* Total de jogos
* Vitórias da IA e do humano
* Empates
* Progresso de aprendizado durante o treinamento

---

## 🧪 Lógica de Treinamento

* A IA joga contra si mesma com política epsilon-greedy
* Armazena transições (estado, ação, recompensa, novo estado, terminal)
* Executa backpropagation a cada jogo com um minibatch de exemplos

---

## 🎯 Sistema de Recompensas

| Situação      | Recompensa |
| ------------- | ---------- |
| Vitória       | +10        |
| Derrota       | -10        |
| Empate        | +1         |
| Jogo continua | 0          |

---

## 📦 Futuras Extensões

* Double DQN
* Prioritized Experience Replay
* Histórico visual de partidas
* Estatísticas de loss e Q-values
* Integração multiplayer online

---

## 👨‍💻 Autor

Desenvolvido por **Nikz**

> Deep Learning Research & Applications

