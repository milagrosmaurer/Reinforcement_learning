import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from .utiles import DQN
from collections import namedtuple
from pypdevs.infinity import INFINITY
from pypdevs.DEVS import AtomicDEVS
import numpy as np

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

class OptimizadorState:
    def __init__(self):
        self.policy_pesos = None
        self.sigma = INFINITY

class Optimizador(AtomicDEVS):
    def __init__(self, seed, nombre_archivos,name="Optimizador"):
        AtomicDEVS.__init__(self, name = name)
        torch.manual_seed(seed)
        self.state = OptimizadorState()
        self.policy_net = DQN(2,3).to(device = device)
        self.target_net = DQN(2,3).to(device = device)
        self.state.policy_pesos = self.policy_net.state_dict()
        self.gamma = 0.99
        self.lr = 1e-3
        self.pasos_actualizar = 500
        self.pasos_optimizar = 4
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr = self.lr)
        self.rng = np.random.default_rng(seed= seed)
        self.criterion = nn.SmoothL1Loss()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.BATCH = self.addInPort("BATCH")
        self.TRANSICION = self.addInPort("TRANSICION")
        self.PESOS = self.addOutPort("PESOS")
        self.pasos = 0
        self.episodios = 1
        self.n_actualizaciones = 0
        self.ventana = 100
        self.pasos_ep = 0
        self.retornos = []
        self.mejor_recompensa = -float("inf")
        self.nombre_archivos = nombre_archivos
        
        
    def intTransition(self):
        '''
        Luego de optimizar los pesos de la red en la delta externa, se pasiva esperando
        más trayectorias del batch.
        '''
        self.state.sigma = INFINITY
        return self.state

    def extTransition(self, inputs):
        '''
        El batch no está preparado para que pase los datos directamente a la red
        los tengo que convertir en tensores.
        '''
        transiciones = inputs.get(self.BATCH)
        transition = inputs.get(self.TRANSICION)

        if transition is not None:
            _, _, terminated, truncated = transition
            self.pasos += 1
            self.state.sigma = INFINITY
            if terminated or truncated:
                self.episodios += 1
                self.retornos.append(-self.pasos_ep)
                if -self.pasos_ep > self.mejor_recompensa:
                    self.mejor_recompensa = -self.pasos_ep
                    torch.save(self.policy_net.state_dict(), f"mejor_modelo_{self.nombre_archivos}.pth")
                self.pasos_ep = 0
            else:
                self.pasos_ep += 1
            

        if transiciones is not None:

            self.n_actualizaciones += 1
            states, actions, rewards, next_states, terminals = transiciones
            q = self.policy_net(states).gather(1, actions).squeeze(1)
            with torch.no_grad():
                next_v = self.target_net(next_states).max(1).values
                target = rewards + self.gamma * next_v * (1.0 - terminals)
            
            loss = self.criterion(q, target)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
            self.optimizer.step()
            self.state.policy_pesos = self.policy_net.state_dict()

            if self.pasos % 5000 == 0:
                torch.save(self.policy_net.state_dict(), 'modelo_pesos.pth')

            self.state.policy_pesos = self.policy_net.state_dict() 
            self.state.sigma = 0.0
            if self.n_actualizaciones % self.pasos_actualizar == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

        return self.state

    def outputFnc(self):
        return {self.PESOS: self.state.policy_pesos}

    def timeAdvance(self):
        return self.state.sigma