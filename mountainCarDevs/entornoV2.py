import numpy as np
from pypdevs.DEVS import AtomicDEVS
from pypdevs.infinity import INFINITY
from collections import namedtuple
import json

FORCE = 0.001
GRAVITY = 0.0025
CUM_REWARD_PER_EPISODE = []

from utiles import TransicionDatos

class EntornoV2State:
    def __init__(self, pos = -0.5, vel = 0.0, sigma = 1.0, terminated = False, truncated = False, reward = -1):
        self.pos = pos
        self.vel = vel
        self.sigma = sigma
        self.terminated = terminated
        self.truncated = truncated
        self.reward = reward

class EntornoV2(AtomicDEVS):
    def __init__(self, name = "EntornoV2"):
        AtomicDEVS.__init__(self,name=name)
        self.state = EntornoV2State()
        self.OBSERVACION = self.addOutPort("OBSERVACION")
        self.ACCION = self.addInPort("ACCION")
        self.episodios = 1
        self.pasos = 0
        self.cum_reward = 0

    def extTransition(self, inputs):
        '''
        A partir de la acción que toma del agente, actualiza su estado.
        '''
        accion = inputs.get(self.ACCION)
        vel_vie = self.state.vel
        pos_vie = self.state.pos
        self.pasos += 1
        self.cum_reward += self.state.reward
        
        vel = vel_vie + (accion - 1)*FORCE - np.cos(3 * pos_vie)*GRAVITY
        pos = pos_vie + vel
        self.state.vel = np.clip(vel,-0.07, 0.07 )
        self.state.pos = np.clip(pos, -1.2, 0.6)
        self.state.truncated = True if self.pasos >= 200 else False
        self.state.terminated = True if self.state.pos >= 0.6 else False
        if self.state.terminated or self.state.truncated:
            print(f" #### El episodio {self.episodios} llegó a su fin con un total de {self.cum_reward} de recompensa acumulada ####")
            #el episodio llegó a su fin, aumento en uno la cantidad de episodios
            self.episodios += 1
            self.pasos = 0

        self.state.sigma = 1.0

        return self.state

    def intTransition(self):

        if self.state.truncated or self.state.terminated:
            self.state = EntornoV2State()
            CUM_REWARD_PER_EPISODE.append(self.cum_reward)
            if self.episodios % 10 == 0:
                with open("datos.json", "w") as f:
                    json.dump(CUM_REWARD_PER_EPISODE, f)
            self.cum_reward = 0
        else:
            #se pasiva el atómico hasta que le llega una nueva acción
            self.state.sigma = INFINITY
        return self.state

    def timeAdvance(self):
        return self.state.sigma

    def outputFnc(self):
        transicion = TransicionDatos(state=[self.state.pos, self.state.vel],
                                    action = None,
                                    next_state=[None, None],
                                    reward= self.state.reward,
                                    truncated= self.state.truncated,
                                    terminated= self.state.terminated)

        #print(f"######### Se ejecuta la transición del entorno con valores: ")
        #print(f"Estado: {transicion.state} | Acción: {transicion.action} | Siguiente estado: {transicion.next_state}")
        #print(f"Reward: {transicion.reward} | Truncated: {transicion.truncated} | Terminated: {transicion.terminated}")
        #print("##########")
        return {self.OBSERVACION : transicion}
