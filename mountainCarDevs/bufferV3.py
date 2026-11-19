from pypdevs.DEVS import AtomicDEVS
from pypdevs.infinity import INFINITY
import torch
import numpy as np

class BufferV3State():
    def __init__(self, capacity, n_obs):
        self.capacity = capacity
        self.states = np.zeros((capacity, n_obs), dtype = np.float32)
        self.next_states = np.zeros((capacity, n_obs), dtype=np.float32)
        self.actions = np.zeros((capacity, 1), dtype = np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.terminals = np.zeros(capacity, dtype = np.float32)
        self.size = 0
        self.sigma = INFINITY


class BufferV3(AtomicDEVS):
    def __init__(self, capacity, n_obs, seed = None,name = 'BufferV3'):
        AtomicDEVS.__init__(self, name = name)
        self.state = BufferV3State(capacity=capacity, n_obs=n_obs)
        self.idx = 0
        self.rng = np.random.default_rng(seed)
        self.pasos = 0
        self.episodios = 0
        self.idx_prev = 0
        self.last_complete = 0
        self.M = 4
        self.pasos_desde_batch = 0

        self.TRANSICION = self.addInPort("TRANSICION")
        self.ACCION = self.addInPort("ACCION")
        self.BATCH = self.addOutPort("BATCH")

    def intTransition(self):
        self.state.sigma = INFINITY
        return self.state

    def extTransition(self, inputs):

        transicion = inputs.get(self.TRANSICION)
        accion = inputs.get(self.ACCION)

        if transicion is not None:
            estado, recompensa, terminado, truncacion = transicion
            #Hay que agregar lo que viene de parte del agente
            if self.pasos > 0:
                p = self.idx_prev
                self.state.next_states[p] = estado
                self.state.rewards[p] = recompensa
                self.state.terminals[p] = 1.0 if terminado else 0.0
                self.last_complete = p
                self.state.size = min(self.state.size + 1, self.state.capacity)   
                
            if terminado or truncacion:
                self.pasos = 0
            else:
                self.state.states[self.idx] = estado
                self.idx_prev = self.idx
                self.idx = (self.idx + 1) % self.state.capacity
                self.pasos += 1
            self.state.sigma = INFINITY

            
        if accion is not None and self.pasos > 0:
            self.state.actions[self.idx_prev] = accion
            #Este es el valor de M: samplear un batch cada M acciones.
            #(llegan entradas cada 1.0) y el timeout nunca se cumpliría.
            self.pasos_desde_batch += 1
            if self.state.size >= 128 and self.pasos_desde_batch >= self.M:
                self.pasos_desde_batch = 0
                self.state.sigma = 0.0
            else:
                self.state.sigma = INFINITY
        return self.state

    def outputFnc(self):
        n = min(self.state.capacity - 1, self.state.size)
        ids = (self.last_complete - self.rng.integers(0, n, 128)) % self.state.capacity

        return {self.BATCH: 
                (torch.from_numpy(self.state.states[ids]),
                torch.from_numpy(self.state.actions[ids]),
                torch.from_numpy(self.state.rewards[ids]),
                torch.from_numpy(self.state.next_states[ids]),
                torch.from_numpy(self.state.terminals[ids]))}

    def timeAdvance(self):
        return self.state.sigma



