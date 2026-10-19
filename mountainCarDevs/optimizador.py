import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from utiles import DQN
from collections import namedtuple
from pypdevs.infinity import INFINITY
from pypdevs.DEVS import AtomicDEVS
import numpy as np
import time

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
from utiles import TransicionDatos

class OptimizadorState:
    def __init__(self):
        self.policy_pesos = None
        self.sigma = INFINITY

class Optimizador(AtomicDEVS):
    def __init__(self, name="Optimizador"):
        AtomicDEVS.__init__(self, name = name)
        self.state = OptimizadorState()
        self.target_net = DQN(2,3).to(device = device)
        self.policy_net = DQN(2,3).to(device = device)
        self.state.policy_pesos = self.policy_net.state_dict()
        self.gamma = 0.99
        self.lr = 3e-4
        self.pasos_actualizar = 10 #TODO pasa al buffer
        self.pasos_optimizar = 4 #TODO esto se tendría que pasar al buffer
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr = self.lr)
        self.rng = np.random.default_rng()
        self.criterion = nn.SmoothL1Loss()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.BATCH = self.addInPort("BATCH")
        self.TRANSICION = self.addInPort("TRANSICION")
        self.PESOS = self.addOutPort("PESOS")
        self.pasos = 0
        self.episodios = 1
        

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
            self.pasos += 1
            if transition.terminated or transition.truncated:
                self.episodios += 1
                self.pasos = 0 #se reinicia el contador de pasos en cada episodio
                self.sigma = INFINITY

        if transiciones is not None:
            
            self.state.policy_pesos = self.policy_net.state_dict() 
            self.state.sigma = 0.0

            if self.pasos % self.pasos_optimizar == 0:
                if len(transiciones) < 128:
                    return self.state
                else:
                    print(f'Actualizando en paso {self.pasos} en episodio {self.episodios}')
                    time_inicial = time.time()
                    batch = TransicionDatos(
                        state=[t.state for t in transiciones],
                        action=[t.action for t in transiciones],
                        next_state=[t.next_state for t in transiciones],
                        reward=[t.reward for t in transiciones],
                        terminated=[t.terminated for t in transiciones],
                        truncated=[t.truncated for t in transiciones]
                    )
                    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                            batch.next_state)))
                    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
                    state_batch = torch.cat(batch.state)
                    action_batch = torch.cat(batch.action)
                    reward_batch = torch.cat(batch.reward)

                    state_action_values = self.policy_net(state_batch).gather(1, action_batch)
                    next_state_values = torch.zeros(128)

                    with torch.no_grad():
                        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values

                    expected_action_values = (next_state_values * self.gamma) + reward_batch
                    loss = self.criterion(state_action_values, expected_action_values.unsqueeze(1))
                    self.optimizer.zero_grad()
                    loss.backward()

                    torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
                    self.optimizer.step()
                    self.state.policy_pesos = self.policy_net.state_dict()
                    time_final = time.time()
                    print(f'El optimizados tarda {(time_final - time_inicial):2f} en actualizar')

            if self.pasos % self.pasos_actualizar == 0:
                policy_net_state_dict = self.policy_net.state_dict()
                self.target_net.load_state_dict(policy_net_state_dict)

            if self.pasos % 100 == 0:
                torch.save(self.policy_net.state_dict(), f'DQN_model_DEVS_v2.pth')
        
        return self.state

    def outputFnc(self):
        return {self.PESOS: self.state.policy_pesos}

    def timeAdvance(self):
        return self.state.sigma