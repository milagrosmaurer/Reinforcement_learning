from pypdevs.DEVS import AtomicDEVS
from pypdevs.DEVS import CoupledDEVS
from .agente import Agente
from .optimizador import Optimizador
from .entornoV2 import EntornoV2
from pypdevs.simulator import Simulator
from .bufferV3 import BufferV3
import time



#AGENTE: IN -> TRANSITION_INPUT, PESOS_INPUT
#        OUT -> ACCION
#BUFFER: IN -> TRANSITION_INPUT, ACTION_INPUT
#        OUT -> BATCH
#OPTIMIZADOR: IN -> TRANSITION_INPUT, BATCH_INPUT
#             OUT -> PESOS
#ENTORNO: IN -> ACTION
#         OUT -> OBSERVATION
#

class MountainCarDevs(CoupledDEVS):
    def __init__(self, nombre_archivos:str,name: str = "MountainCarDevs", seed: int = 42):
        CoupledDEVS.__init__(self, name = name)
        self.entorno = self.addSubModel(EntornoV2(nombre_archivos,seed = seed))
        self.agente = self.addSubModel(Agente(nombre_archivos, seed = seed))
        self.buffer = self.addSubModel(BufferV3(seed = seed, n_obs=2, capacity=10000))
        self.optimizador = self.addSubModel(Optimizador(nombre_archivos=nombre_archivos, seed = seed))

        #El mensaje del entorno hacia los distintos atómicos
        self.connectPorts(self.entorno.OBSERVACION, self.agente.TRANSICION)
        self.connectPorts(self.entorno.OBSERVACION, self.buffer.TRANSICION)
        self.connectPorts(self.entorno.OBSERVACION, self.optimizador.TRANSICION)

        #El batch le manda al optimizador
        self.connectPorts(self.buffer.BATCH, self.optimizador.BATCH)

        #El optimizador le mando los pesos al entorno
        self.connectPorts(self.optimizador.PESOS, self.agente.PESOS)

        #El agente le manda la acción al entorno y al buffer
        self.connectPorts(self.agente.ACCION, self.buffer.ACCION)
        self.connectPorts(self.agente.ACCION, self.entorno.ACCION)

    def select(self, imm_children):
    
        for m in imm_children:
            if isinstance(m, BufferV3):
                return m
        return imm_children[0]

class mountainCarDevsAprendido(CoupledDEVS):
    def __init__(self, nombre_archivos: str, name: str = 'AprendizajeMountainCar', seed: int = 42):
        CoupledDEVS.__init__(self, name = name)
        self.entorno = self.addSubModel(EntornoV2(nombre_archivos,seed = seed))
        self.agente = self.addSubModel(Agente(nombre_archivos, seed = seed))

        self.connectPorts(self.entorno.OBSERVACION, self.agente.TRANSICION)

        self.connectPorts(self.agente.ACCION, self.entorno.ACCION)

    def select(self, imm_children):
        
            for m in imm_children:
                if isinstance(m, EntornoV2):
                    return m
            return imm_children[0]

def main(nombre_archivos: str, name: str, pasos: float, seed: int = 42,entrenamiento = False):
    #TODO agregar carpeta de guardado de pesos y datos.

    if entrenamiento:
        mountainCar = MountainCarDevs(nombre_archivos= nombre_archivos, name = name, seed= seed)
    else:
        mountainCar = mountainCarDevsAprendido(nombre_archivos=nombre_archivos, name = name, seed=seed)

    
    sim = Simulator(mountainCar)
    sim.setTerminationTime(pasos)
    sim.setClassicDEVS()
    sim.setMessageCopy("none")
    tiempo_inicial = time.time()
    sim.simulate()
    tiempo_final = time.time()
    print(f'El tiempo que tardó devs fue: {tiempo_final - tiempo_inicial}')
        
