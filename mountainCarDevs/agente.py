#TODO este uso episodios para el decaimiento del epsilon
import numpy as np
from pypdevs.DEVS import AtomicDEVS
from pypdevs.infinity import INFINITY
import random
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from utiles import DQN
from collections import namedtuple

from utiles import TransicionDatos
#el buffer y el entorno
#conecto el entorno con el optimizador para avisarle que termino
#que cada uno cuente por su cuenta

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

class AgenteState:
    def __init__(self, action = None, sigma = INFINITY):
        #TODO cambiar a None
        self.action = action
        self.sigma = sigma

class Agente(AtomicDEVS):
    def __init__(self, name = "agente"):
        AtomicDEVS.__init__(self, name=name)
        self.state = AgenteState()
        
        self.policy_net = DQN(2,3).to(device = device)
        self.policy_net.eval()
        self.rng = np.random.default_rng()
        self.PESOS = self.addInPort("PESOS")
        self.TRANSICION = self.addInPort("TRANSICION")
        self.ACCION = self.addOutPort("ACCION")
        self.epsilon = 1.0
        self.pasos = 0
        self.eps_end = 0.001
        self.eps_start = 0.99
        self.eps_decay = 4000
        self.episodios = 1

    def intTransition(self):
        self.state.sigma = INFINITY
        return self.state

    def extTransition(self, inputs):
        
        pesos = inputs.get(self.PESOS)
        transicion = inputs.get(self.TRANSICION)
        
        if pesos is not None:
            #Cambiamos el estado de nuestros pesos, y pasivamos el atómico para la próxima transición
            self.policy_net.load_state_dict(pesos)
            self.sigma = INFINITY

        if transicion is not None:
            estado_actual = transicion.state
            terminated = transicion.terminated
            truncated = transicion.truncated
            self.pasos += 1
            if terminated or truncated:
                self.pasos = 0
                self.episodios += 1
                self.state.sigma = INFINITY
            else:
                self.state.sigma = 0.0

            self.epsilon = max(
                self.eps_end,
                self.eps_start - (self.eps_start - self.eps_end) * (self.episodios / self.eps_decay)
            )
            nro_random = random.random()
            if nro_random < self.epsilon:
                with torch.no_grad():
                    self.state.action = torch.tensor([[self.rng.integers(0,3)]], dtype = torch.long, device=device)
            else:
                estado_actual_tensor = torch.tensor(estado_actual, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    self.state.action = self.policy_net(estado_actual_tensor).max(1).indices.view(1,1)
       
        return self.state

    def outputFnc(self):
        return {self.ACCION : self.state.action.item()}

    def timeAdvance(self):
        return self.state.sigma

        

                

