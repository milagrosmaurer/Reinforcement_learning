import numpy as np
import gymnasium as gym
import json

import json
import numpy as np

import matplotlib.pyplot as plt
import time

class SetStartWrapper(gym.Wrapper):
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.env.unwrapped.state = [-1, 0.0] # -------------> Para cambiar la condición inicial acáaaaa!!!!
        return self.env.unwrapped.state, info

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
    
class QLearningAgente:
    def __init__(self, pos_space, vel_space, num_actions, alpha = 0.01, gamma = 0.9, epsilon = 1.0):
        self.pos_space = pos_space
        self.vel_space = vel_space
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = np.random.default_rng()
        self.q = np.zeros((len(pos_space) + 1, len(vel_space) + 1, num_actions))

    def digitize_state(self,state):

        state_pos = np.digitize(state[0], self.pos_space)
        state_vel = np.digitize(state[1], self.vel_space)

        return (state_pos, state_vel)

    def epsilon_linear(self, episode, epsilon_start=1.0, epsilon_end=0.0, decay_episodes=30000):
        self.epsilon = max(epsilon_end, epsilon_start - (epsilon_start - epsilon_end) * (episode / decay_episodes))

    def elegir_accion(self,state_disc):

        if self.rng.random() < self.epsilon:
            action = self.rng.integers(0, self.num_actions) #samplea una acción cualquiera
        else:
            action = np.argmax(self.q[state_disc])
        return action
    
    def actualizar_q(self, experiencias):
        rewards = 0
        for state, action, reward, new_state in experiencias:

            #state_pos, state_vel = self._digitize_state(state)
            state_disc = self.digitize_state(state)
            #new_state_pos, new_state_vel = self._digitize_state(new_state)
            new_state_disc = self.digitize_state(new_state)

            #self.q[state_pos, state_vel, action] = self.q[state_pos, state_vel, action] + self.alpha*(
             #   reward + self.gamma * np.max(self.q[new_state_pos,new_state_vel,:]) - self.q[state_pos, state_vel, action]
            #)
            self.q[state_disc + (action,)] = self.q[state_disc + (action,)] + self.alpha*(
                            reward + self.gamma * np.max(self.q[new_state_disc]) - self.q[state_disc + (action,)]
                    )

            rewards += reward
        return rewards
#la lambda no cambia el estado


def plot_reward(num_episodes, rewards, tiempo,window = 100):
    tiempo = time.time()
    mean_rewards = np.zeros(num_episodes)
    for t in range(num_episodes):
        mean_rewards[t] = np.mean(rewards[max(0,t - window):t + 1]) #mean rewarsa de los últimos 100 espisodios
    plt.plot(mean_rewards)
    plt.ylabel('recompensas promedio')
    plt.xlabel('episodios')
    plt.savefig(f'imagenes\mountainCar\q_{tiempo}.png')


def recolectar_experiencias(env, agente, pos_space, vel_space):

    terminated = False
    truncated = False

    state,_ = env.reset()

    experience = []


    while not terminated and not truncated:

        state_disc = agente.digitize_state(state)

        action = agente.elegir_accion(state_disc)
        

        new_state, reward, terminated, truncated, info = env.step(action)

        experience.append((state, action, reward, new_state))

        state = new_state

    return experience

            
def train(num_episodes, divisiones):

    tiempo = time.time()

    env = gym.make('MountainCar-v0')

    pos_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0], divisiones)
    vel_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], divisiones)
    num_actions = env.action_space.n
    agente = QLearningAgente(pos_space=pos_space, vel_space=vel_space,num_actions=num_actions)
    rewards_per_episode = []
    for i in range(num_episodes):
        agente.epsilon_linear(i, decay_episodes=10000)
        experience = recolectar_experiencias(env, agente,pos_space, vel_space)
        rewards = agente.actualizar_q(experiencias=experience)
        rewards_per_episode.append(rewards)
        if i%100 == 0:
            print(f'Entrenando episodio {i} con reward {rewards}')

    with open(f"matrizQ_{tiempo}.json", "w") as f:
        json.dump(agente.q, f,cls=NumpyEncoder)
    env.close()

    plot_reward(num_episodes, rewards_per_episode, tiempo)


def digitize_state(state, pos_space, vel_space):

    state_pos = np.digitize(state[0], pos_space)
    state_vel = np.digitize(state[1], vel_space)

    return state_pos, state_vel    
        
def ver(archivo_q):

    with open(archivo_q,"r") as f:
        q = np.array(json.load(f))

    env = gym.make('MountainCar-v0', render_mode = 'human')
    pos_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 20)
    vel_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 20)

    for i in range(20):

        state, _ = env.reset()

        terminated = False
        truncated = False

        while not terminated and not truncated:

            state_disc = digitize_state(state, pos_space, vel_space)

            action = np.argmax(q[state_disc])
            state, reward, terminated, truncated, info = env.step(action)


if __name__ == "__main__":
    #train(30000, 20)
    ver("matrizQ_1785359364.6335158.json")


    
    