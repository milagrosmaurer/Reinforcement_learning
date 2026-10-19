from pypdevs.DEVS import AtomicDEVS
from pypdevs.infinity import INFINITY
from collections import deque, namedtuple
import random
from utiles import chequeo
import torch

'''
El buffer recibe información de dos atómicos, el del agente y el del entorno.
El del entorno le da la siguiente transición (estado_viejo, estado_nuevo, reward, terminated y truncated)
El entorno le da la acción que corresponde al cambio entre el estado_viejo y el estado_nuevo
Si al Buffer le llega un state terminal, no hace append de la transición y se queda esperando.
'''
from utiles import TransicionDatos

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

class BufferState:
    def __init__(self):
        self.capacity = 10000
        self.memoria = deque([], self.capacity)
        self.sigma = INFINITY
        self.pasos = 0
        self.episodios = 1
        
    def _len_memoria(self):
        return len(self.memoria)
        

class Buffer(AtomicDEVS):
    def __init__(self, name="Buffer"):
        AtomicDEVS.__init__(self, name=name)
        self.state = BufferState()
        self.batch_size = 128
        self.ACCION = self.addInPort("ACCION")
        self.TRANSICION = self.addInPort("TRANSICION")
        self.BATCH = self.addOutPort("BATCH")
        self.transicion_append = TransicionDatos()

    def intTransition(self):
        self.state.sigma = INFINITY #me quedo pasivado hasta que venga la próxima transición
        return self.state

    def extTransition(self, inputs):
        transicion = inputs.get(self.TRANSICION)
        #transicion.next_state = None if transicion.terminated else torch.tensor(transicion.next_state, dtype=torch.float32).unsqueeze(0)
        accion = inputs.get(self.ACCION)

        if transicion is not None:
            transicion.state = torch.tensor(transicion.state, dtype=torch.float32, device=device).unsqueeze(0)
            transicion.reward = torch.tensor([transicion.reward], dtype = torch.float32, device=device)
            if self.state.pasos == 0:
                #Es la primer transición, así que no la tengo que agregar en append dos veces
                self.transicion_append.state = transicion.state
            else:
                self.transicion_append.next_state = transicion.state
                self.transicion_append.reward = transicion.reward
                self.transicion_append.truncated = transicion.truncated
                self.transicion_append.terminated = transicion.terminated
                self.state.memoria.append(self.transicion_append)
                self.transicion_append = TransicionDatos(state=transicion.state)

            self.state.pasos += 1
            if transicion.terminated or transicion.truncated:
                self.state.episodios += 1
                self.transicion_append.action = torch.tensor([[0]], dtype = torch.long, device=device)
                self.transicion_append.reward = torch.tensor([transicion.reward], dtype = torch.float32, device=device)
                print(f'# Se agregó la transición {self.transicion_append} al buffer en el paso {self.state.pasos}')
                self.state.memoria.append(self.transicion_append)
                self.state.pasos = 0
                self.transicion_append =TransicionDatos()

            self.state.sigma = INFINITY
            
        if accion is not None:

            accion = torch.tensor([[accion]], dtype = torch.long, device=device)
            self.transicion_append.action = accion
            
            if len(self.state.memoria) >= 128:
                self.state.sigma = 0.0
            else:
                self.state.sigma = INFINITY

        return self.state

    def timeAdvance(self):
        return self.state.sigma

    def outputFnc(self):
        batch = random.sample(self.state.memoria, min(self.batch_size, self.state._len_memoria()))
        return {self.BATCH : batch} 

        
        