import numpy as np
from pypdevs.DEVS import AtomicDEVS
from pypdevs.infinity import INFINITY
import time
import pickle
import json
import math

# Create a JSON Encoder class
class json_serialize(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

#TODO guardar datos, ver tema epsilon, para mi puede ser que eso se este calculando mal.
#ver desde que posición empieza, quizá arranca mal o muy poco distribuido lo cual seria raro.

FORCE = 0.001
GRAVITY = 0.0025

class EntornoV2State:
    def __init__(self, rng,vel = 0.0, sigma = 0.0, terminated = False, truncated = False, reward = -1):
        #self.pos = rng.uniform(low=-0.6, high=-0.4)
        self.pos = -0.5
        self.vel = vel
        self.sigma = sigma
        self.terminated = terminated
        self.truncated = truncated
        self.reward = reward      

class EntornoV2(AtomicDEVS):

    def __init__(self,nombre_archivos, seed,name = "EntornoV2"):
        AtomicDEVS.__init__(self,name=name)
        self.rng = np.random.default_rng(seed)
        self.state = EntornoV2State(self.rng)
        self.OBSERVACION = self.addOutPort("OBSERVACION")
        self.ACCION = self.addInPort("ACCION")
        self.episodios = 1
        self.pasos = 0
        self.cum_reward = 0
        self.max_pos = self.state.pos
        self.nombre_archivo = nombre_archivos
        self.rewards = []
        self.start_pos = []
        self.max_positions = []
        self.start_pos.append(self.state.pos)
        self.transiciones = []
        self.estados = []
        self.velocidades = []
        
    def extTransition(self, inputs):
        '''
        A partir de la acción que toma del agente, actualiza su estado.
        '''
        accion = inputs.get(self.ACCION)
        vel_vie = self.state.vel
        pos_vie = self.state.pos
        self.pasos += 1
        self.cum_reward += self.state.reward

        self.estados.append(pos_vie)
        self.velocidades.append(vel_vie)

        #Actualizamos velocidad
        vel = vel_vie + ((accion - 1) * FORCE + math.cos(3 * pos_vie) * (-GRAVITY))
        vel = np.clip(vel,-0.07, 0.07)

        #Actualizamos posicion
        pos = pos_vie + vel
        pos = np.clip(pos, -1.2, 0.6)

        #Me guardo la mayor posicion alcanzada en un episodio
        self.max_pos = max(self.max_pos, self.state.pos)
        self.state.pos = pos
        self.state.vel = 0.0 if (self.state.pos == -1.2 and self.state.vel < 0) else vel

        self.transiciones.append((np.array([pos_vie, vel_vie], dtype = np.float32), accion, np.array([self.state.pos, self.state.vel], dtype = np.float32)))
        
        self.state.truncated = True if self.pasos >= 200 else False
        self.state.terminated = True if (self.state.pos >= 0.5 and self.state.vel >= 0) else False
        if self.state.terminated or self.state.truncated:
            self.episodios += 1
            self.pasos = 0
            self.max_positions.append(self.max_pos)
            self.max_pos = -1.2
            if self.episodios % 50 == 0:
                print(f'El episodio {self.episodios} terminó con una recompensa acumulada de {self.cum_reward}')
        self.state.sigma = 1.0

        return self.state

    def intTransition(self):

        if self.state.truncated or self.state.terminated:
            self.rewards.append(self.cum_reward)
            if len(self.rewards) % 50 == 0:
                ult = np.array(self.rewards[-100:])
                print(f"ep {len(self.rewards)}: prom100={ult.mean():.1f}  mejor={ult.max():.0f}  "
                    f"episodios en -200={np.mean(ult == -200)*100:.0f}%")
            self.cum_reward = 0
            self.state = EntornoV2State(self.rng)
            self.start_pos.append(self.state.pos)
            self.state.sigma = 1.0
            if self.episodios % 2 == 0:
                with open(f"resultados/resultados_{self.nombre_archivo}.json", "w") as archivo:
                    #json.dump({"rewards": self.rewards, "posicion_start": self.start_pos, "max_pos": self.max_positions}, archivo)
                    json.dump({"transiciones": self.transiciones, 'posiciones': self.estados, 'velocidades': self.velocidades}, archivo, cls = json_serialize)
        else:
            self.state.sigma = INFINITY
        return self.state

    def timeAdvance(self):

        return self.state.sigma

    def outputFnc(self):

        transicion = ([self.state.pos, self.state.vel], self.state.reward, self.state.terminated, self.state.truncated)
        return {self.OBSERVACION : transicion}
