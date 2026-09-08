import numpy as np
from pypdevs.DEVS import AtomicDEVS
from pypdevs.simulator import Simulator
from pypdevs.infinity import INFINITY
from pypdevs.DEVS import CoupledDEVS
import math
import random
from renderMC import RenderMC
import torch
from torch import nn
import torch.nn.functional as F
from utiles import DQN, ReplayMemory
from collections import namedtuple
'''
Si los dos tienen sigma infinito se termina la simulación
Y que expulse la terminación, truncación y reward
Agregar al átomico del agente la replay memory como un atributo - lo puedo inicializar afuera para no perder las
trayectorias entre episodios.
Cómo determino de qué puerto vino cada input?? TODO preguntar

'''
FORCE = 0.001
GRAVITY = 0.0025

'''
Si tengo sigma como parte de mi estado no hace falta especificar la time advance
solo con return self.state.sigma
Cuando llega un evento externo se ejecuta la delta_ext
 

Cuando se cumple el tiempo en la agenda del próximo evento:

'''

Transition = namedtuple('Transition',
                        ['state', 'action', 'next_state', 'reward', 'terminated', 'truncated'])

class MountainCarState:
    def __init__(self, pos, vel = 0.0, sigma = 0.0, terminated = False, truncated = False,reward = -1):
        self.pos_vie = 0.0
        self.vel_vie = 0.0
        self.pos = pos
        self.vel = vel
        self.sigma = sigma
        self.terminated = terminated
        self.truncated = truncated
        self.pasos = 0
        self.reward = -1
        self.action = None
        
class MountainCar(AtomicDEVS):
    def __init__(self, name = "MountainCar"):
        AtomicDEVS.__init__(self, name)
        self.state = MountainCarState(pos = -0.5, sigma= 0.0)
        #self.state = MountainCarState(pos = -0.5, vel = 0.0, sigma = 1.0)
        #TODO acordate que hay que clipear la posicion y la velocidad para
        #que no se vayan de rango pos = -1.2, 0.6 y vel = -0.07, 0.07

        self.OBSERVED = self.addOutPort("OBSERVED")
        self.ACTION = self.addInPort("ACTION")

        self.positions = []
        self.velocities = []

    def intTransition(self):
        self.state.sigma = INFINITY
        #me agendo un evento para ser expulsado en tiempo 1, el estado en si cambia en la ext
        #se pausa el estado porque esta a la espera de que venga una nueva acción???????
        #en realidad se
        
        return self.state

    def extTransition(self, inputs):
        #Guardamos el estado anterior
        pos_act = self.state.pos
        vel_act = self.state.vel

        #Valor dentro de {0,1,2}
        action = inputs.get(self.ACTION) 

        #Ecuaciones para determinar el nuevo estado
        vel_nue = vel_act + (action - 1)*FORCE - np.cos(3 * pos_act)*GRAVITY
        pos_nue = pos_act + vel_nue
        vel_clip = np.clip(vel_nue, -0.07, 0.07)
        pos_clip = np.clip(pos_nue, -1.2, 0.6)

        #Cambiamos el estado
        self.state.pos = pos_clip
        self.state.vel = vel_clip 
        self.state.sigma = 0.0 #Queremos que se ejecute en el instante
        self.state.pos_vie = pos_act
        self.state.vel_vie = vel_act
        self.state.pasos += 1 #Aumentamos en uno la cantidad de pasos hechos
        self.state.reward = -1 #Es la misma en todos los pasos, pero para hacerlo más general
        self.state.terminated = True if self.state.pos >= 0.6 else False
        self.state.truncated = True if self.state.pasos >= 200 else False
        self.state.action = action
        self.positions.append(self.state.pos)
        self.velocities.append(self.state.vel)

        print(f'Se ejecutó la delta externa con los siguientes valores en tiempo {self.time_last}')
        print(f'El estado es posición: {self.state.pos} y la velocidad: {self.state.vel}')
        print(f'El sigma es: {self.state.sigma}') 

        return self.state

    def timeAdvance(self):
        return self.state.sigma
        
    def outputFnc(self):
        state_vie = [self.state.pos_vie, self.state.vel_vie]
        state_nue = [self.state.pos, self.state.vel]
        reward = self.state.reward
        truncated = self.state.truncated
        terminated = self.state.terminated
        action = self.state.action
        
        transicion = Transition(state=state_vie, next_state= state_nue, terminated=terminated,
                               truncated=truncated, reward=reward, action=action)
        
        return {self.OBSERVED: transicion}

class RandomActionState:
    def __init__(self):
        self.sigma = 0.32

class RandomAction(AtomicDEVS):
    def __init__(self, name="AccionRandom"):
        AtomicDEVS.__init__(self, name)
        self.state = RandomActionState()
        self.ACTION_OBSERVED = self.addOutPort("ACTION_OBSERVED")

    def timeAdvance(self):
        return self.state.sigma

    def outputFnc(self):
        action_random = random.randint(0,2)
        return {self.ACTION_OBSERVED: action_random}

    def intTransition(self):
        return self.state


class AccionAprendidaState:
    def __init__(self,action,sigma):
        self.action = action
        self.sigma = sigma

class AccionAprendida(AtomicDEVS):
    def __init__(self, name="AccionAprendida", archivo_politica = "DQN_model_20260819-203228.pth" ):
        AtomicDEVS.__init__(self, name)
        self.state = AccionAprendidaState(action=1, sigma=INFINITY)
        self.memoria = ReplayMemory(capacity=10000)
        
        self.STATE = self.addInPort("STATE")
        self.ACTION_CHOOSE = self.addOutPort("ACTION_CHOOSE")
        self.policy_net = DQN(2, 3)
        self.policy_net.load_state_dict(torch.load(archivo_politica,weights_only=True))
        self.policy_net.eval()

    def extTransition(self, inputs):
        in_state = inputs.get(self.STATE)
        estado_actual = in_state.next_state
        state = torch.tensor(estado_actual,dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = self.policy_net(state).max(1).indices.view(1, 1)
        self.state.action = action.item()
        #self.state.sigma = self.state.sigma - self.elapsed
        estado_terminacion = in_state.terminated
        estado_truncacion = in_state.truncated
        if estado_truncacion or estado_terminacion:
            #si el estado llegó a algún estado terminal es necesario poner el sigma en infinito así se frena la simulación
            self.state.sigma = INFINITY
        else:
            self.state.sigma = 1.0
        print(len(self.memoria))
        self.memoria.push(in_state)
        return self.state

    def intTransition(self):
        self.state.sigma = INFINITY
        return self.state

    def timeAdvance(self):
        return self.state.sigma

    def outputFnc(self):
        return {self.ACTION_CHOOSE: self.state.action}

class MountainCarRL(CoupledDEVS):
    def __init__(self, name="EntornoCompleto"):

        CoupledDEVS.__init__(self, name=name)
        self.entorno = self.addSubModel(MountainCar())
        #para acción random
        #self.agente = self.addSubModel(RandomAction())
        #self.connectPorts(self.agente.ACTION_OBSERVED, self.entorno.ACTION)

        #para acción aprendida desde política aprendida
        self.agente = self.addSubModel(AccionAprendida())
        self.connectPorts(self.agente.ACTION_CHOOSE, self.entorno.ACTION)
        self.connectPorts(self.entorno.OBSERVED, self.agente.STATE )
        

    def select(self, immChildren):
        #hace el chequeo de que el atomico sea hijo de algun submodelo
        if self.entorno in immChildren:
            return self.entorno
        else:
            return immChildren[0]



if __name__ == '__main__':
    mountainCar = MountainCarRL(name='mountainCar')
    sim = Simulator(mountainCar)
    #La simulación termina cuando el auto llega a la bandera
    #def terminate_whenStateIsReached(clock, model):
     #   return model.mountainCar.entorno.pos >= 0.6
    #sim.setTerminationCondition(terminate_whenStateIsReached)
    sim.setTerminationTime(400.0) #también le podes poner una condición de terminación
    #en mountainCar sería cuando pos == 0.6
    #sim.setVerbose() #si le pones el nombre de un archivo te lo pone ahí, sino te lo imprime en consola, si la comento no imprime nada
    sim.setClassicDEVS()
    sim.simulate()

    animacion = RenderMC("human")
    animacion.set_trajectory(mountainCar.entorno.positions, mountainCar.entorno.velocities)
    animacion.play_trajectory()
    
    
