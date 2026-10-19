from pypdevs.DEVS import AtomicDEVS
from pypdevs.DEVS import CoupledDEVS
from agente import Agente
from bufferV2 import Buffer
from optimizador import Optimizador
from entornoV2 import EntornoV2
from pypdevs.simulator import Simulator

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
    def __init__(self, name = "MountainCarDevs"):
        CoupledDEVS.__init__(self, name = name)
        self.entorno = self.addSubModel(EntornoV2())
        self.agente = self.addSubModel(Agente())
        self.buffer = self.addSubModel(Buffer())
        self.optimizador = self.addSubModel(Optimizador())

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
        if self.entorno in imm_children:
            return self.entorno
       

if __name__ == '__main__':
    mountainCar = MountainCarDevs(name='mountainCar')
    sim = Simulator(mountainCar)
    sim.setTerminationTime(8000000.0)
    sim.setClassicDEVS()
    sim.simulate()

        
