import copy
import json
import math
import os
import time

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn


class ReplayMemory:
    """Buffer circular con arrays preasignados (sin None para estados terminales)."""

    def __init__(self, capacity, n_obs, seed=None):
        self.capacity = capacity
        self.states = np.zeros((capacity, n_obs), dtype=np.float32)
        self.next_states = np.zeros((capacity, n_obs), dtype=np.float32)
        self.actions = np.zeros((capacity, 1), dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.terminals = np.zeros(capacity, dtype=np.float32)  # 1.0 si terminó
        self.idx = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def push(self, state, action, next_state, reward, terminated):
        i = self.idx
        self.states[i] = state
        self.actions[i] = action
        self.next_states[i] = next_state
        self.rewards[i] = reward
        self.terminals[i] = float(terminated)
        self.idx = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        ids = self.rng.integers(0, self.size, batch_size)
        return (torch.from_numpy(self.states[ids]),
                torch.from_numpy(self.actions[ids]),
                torch.from_numpy(self.rewards[ids]),
                torch.from_numpy(self.next_states[ids]),
                torch.from_numpy(self.terminals[ids]))

    def __len__(self):
        return self.size


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super().__init__()
        self.layer1 = nn.Linear(n_observations, 64)
        self.layer2 = nn.Linear(64, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        return self.layer2(x)

class AgenteDQN:
    def __init__(self, n_observations, n_actions,
                 batch_size=128, gamma=0.99, eps_start=1.0,
                 eps_end=0.01, eps_decay=4000, N=500, M=4,
                 lr=1e-3, mem_size=10000, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        self.n_observations = n_observations
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = eps_start
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.pasos_actualizar = N
        self.pasos_optimizar = M
        self.lr = lr
        self.mejor_estado = None  # state_dict del mejor checkpoint (si se evalúa durante el entrenamiento)
        self.n_updates = 0
        self.mejor_estado_prom = None
        
        self.policy_net = DQN(n_observations, n_actions)
        self.target_net = DQN(n_observations, n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.memory = ReplayMemory(mem_size, n_observations, seed)
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr = self.lr)
        self.rng = np.random.default_rng(seed)
        self.criterion = nn.SmoothL1Loss()

    def epsilon_linear(self, episode):
        self.epsilon = max(self.eps_end,
                           self.eps_start - (self.eps_start - self.eps_end) * (episode / self.eps_decay))

    def accion_greedy(self, state):
        """Acción greedy; no consume números aleatorios (no altera la reproducibilidad)."""
        with torch.no_grad():
            return int(self.policy_net(state).argmax(1).item())

    def elegir_accion(self, state):
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.n_actions))
        return self.accion_greedy(state)

    def optimizar(self):
        if len(self.memory) < self.batch_size:
            return
        states, actions, rewards, next_states, terminals = self.memory.sample(self.batch_size)

        # Q(s, a) para la acción tomada
        q = self.policy_net(states).gather(1, actions).squeeze(1)

        # Para estados terminales no hay valor futuro
        with torch.no_grad():
            next_v = self.target_net(next_states).max(1).values
            target = rewards + self.gamma * next_v * (1.0 - terminals)

        loss = self.criterion(q, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        self.n_updates += 1
        if self.n_updates % self.pasos_actualizar == 0:
            self.actualizar_target()
        

    def actualizar_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


def train(num_episodes, nombre, seed=42, guardar=False,
          eval_cada=None, mem_size=10000, ):
    """
    Devuelve (agente, lista_rewards, epsilons, tiempos_episodios).

    eval_cada: si se indica, evalúa greedy cada esa cantidad de episodios y guarda
               en agente.mejor_estado los pesos del mejor checkpoint.
    """
    env = gym.make('MountainCar-v0')
    env_eval = gym.make('MountainCar-v0') if eval_cada else None
    n_actions = env.action_space.n
    n_observations = env.observation_space.shape[0]

    agente = AgenteDQN(n_observations=n_observations,
                       n_actions=n_actions,
                       mem_size=mem_size,
                       seed=seed)

    lista_rewards, epsilons, tiempos_episodios = [], [], []
    mejor_score = -np.inf
    pasos_global = 0

    tiempo_inicial = time.time()

    for i_episode in range(num_episodes):
        t0 = time.time()
        # Solo el primer reset recibe la semilla; los demás continúan la secuencia
        state, _ = env.reset(seed=seed if i_episode == 0 else None)
        agente.epsilon_linear(i_episode + 1)
        epsilons.append(agente.epsilon)

        terminated = truncated = False
        rewards = 0.0

        while not (terminated or truncated):
            
            action = agente.elegir_accion(torch.from_numpy(state).unsqueeze(0))
            observation, reward, terminated, truncated, _ = env.step(action)
            pasos_global += 1
            rewards += reward

            # Si truncó (no terminó) se sigue haciendo bootstrap: terminals=0
            agente.memory.push(state, action, observation, reward, terminated)
            state = observation

            if pasos_global % agente.pasos_optimizar == 0:
                agente.optimizar()
        lista_rewards.append(rewards)

        if mejor_score < rewards:
            mejor_score = rewards
            torch.save(agente.policy_net.state_dict(), f'DQN_model_{nombre}_mejor_score.pth')
            
        tiempos_episodios.append(time.time() - t0)
        if i_episode % 500 == 0:
            print(f'Episodio {i_episode} | recompensa {rewards} | epsilon {agente.epsilon:.3f}')

    tiempo_final = time.time()
            
    env.close()
    tiempo_total = tiempo_final - tiempo_inicial
    if guardar:
        torch.save(agente.policy_net.state_dict(), f'DQN_model_{nombre}.pth')
        if agente.mejor_estado is not None:
            torch.save(agente.mejor_estado, f'DQN_model_{nombre}_mejor_checkpoint.pth')
        with open(f'resultados_{nombre}.json', 'w') as archivo:
            json.dump({'lista_rewards': lista_rewards,
                       'epsilons': epsilons,
                       'tiempos_episodios': tiempo_total}, archivo, indent=4)


