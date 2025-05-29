# 🤖 Jogo da Velha com Aprendizado por Reforço (DQN)

Este projeto implementa um **agente baseado em redes neurais treinado com Deep Q-Learning (DQN)** para jogar **Tic-Tac-Toe (Jogo da Velha)**. A interface gráfica é feita com **Tkinter**, e o modelo de rede neural é desenvolvido com **PyTorch**.

---

> 📘 **Antes de continuar, confira o artigo completo:**  
> [**Redes Neurais Profundas e Deep Q-Learning: Da Teoria à Aplicação Corporativa**](https://github.com/NicollasRezende/NeuralNK/blob/main/RedesNeurais.md)  
> Um mergulho teórico e prático sobre como redes neurais profundas podem ser aplicadas em ambientes corporativos com Deep Q-Learning.

---

## 📂 Estrutura do Projeto

```

📁 projeto/
├── main.py               # Ponto de entrada que inicia a aplicação
├── model.py              # Arquitetura da rede neural (TicTacToeNet)
├── agent.py              # Implementação do agente DQN
├── game.py               # Regras do jogo, verificação de vitória e sistema de recompensa
├── ui.py                 # Interface gráfica com modos de jogo e treinamento
├── tic_tac_toe_model.pth # Arquivo de modelo salvo
├── README.md             # Documentação do projeto

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

### 🔸 `agent.py`: Agente com Deep Q-Learning

Responsável por:

* Tomar decisões usando política epsilon-greedy
* Treinar com minibatches a partir de um buffer de experiência
* Atualizar uma rede de destino para maior estabilidade
* Aplicar heurísticas simples para reforçar estratégias vencedoras

---

### 🔸 `game.py`: Lógica do Jogo

Contém funções para:

* Verificar jogadas válidas
* Aplicar ações no tabuleiro
* Detectar vitórias ou empates
* Calcular a recompensa com base no resultado

---

### 🔸 `ui.py`: Interface Gráfica

Interface interativa com suporte para:

* 🧑 vs agente: humano contra o agente treinado
* agente vs agente: simulação entre dois agentes
* 🎯 Treinar: partidas automáticas entre agentes para treinamento contínuo
* Estatísticas e botão para salvar o modelo

---

## 🚀 Como Executar

### 1. Instalar dependências

```bash
pip install torch numpy
```

### 2. Executar o projeto

```bash
python main.py
```

> Observação: `tkinter` já vem instalado com a maioria das distribuições Python. Em alguns Linux, use `sudo apt install python3-tk`.

---

## 🎮 Modos de Jogo

| Modo             | Descrição                               |
| ---------------- | --------------------------------------- |
| 🧑 vs agente     | Humano joga contra o agente treinado    |
| agente vs agente | Simulação entre dois agentes            |
| 🎯 Treinar       | Agente aprende jogando contra ele mesmo |

---

## 💾 Salvamento de Modelo

Você pode salvar o estado atual do modelo clicando no botão **💾 Salvar Modelo**.
O arquivo será salvo como `tic_tac_toe_model.pth`.

---

## 📊 Estatísticas em Tempo Real

* Total de jogos
* Vitórias do humano e do agente
* Empates
* Progresso do agente durante o treinamento

---

## 🧪 Lógica de Treinamento

* O agente joga contra ele mesmo com política epsilon-greedy
* Cada episódio gera transições (estado, ação, recompensa, próximo estado, final)
* Executa aprendizado por minibatches com backpropagation

---

## 🎯 Sistema de Recompensas

| Situação      | Recompensa |
| ------------- | ---------- |
| Vitória       | +10        |
| Derrota       | -10        |
| Empate        | +1         |
| Jogo em curso | 0          |

---

## 📦 Futuras Extensões

* Double DQN
* Prioritized Experience Replay
* Histórico visual de partidas
* Métricas de loss e Q-values
* Integração com multiplayer online

---

## 👨‍💻 Autor

Desenvolvido por **Nikz**

> Deep Learning Research & Applications
