"""
MountainCar Discreto

Modelo atomic DEVS que cuando recibe una acción desde otro modelo atomic DEVS ejecuta esa acción.
Devuelve una tupla <estado, reward, termination, truncation, info>

Formalismo:

    M_P = <X, S, Y, delta_int, delta_ext, lambda, ta>

    X          = {action_space}                    (sin entradas)
    S          = {observation_space} x R+              (un solo estado + sigma = tiempo restante)
    Y          = <observation_space, N, bool, bool, dict>                     (salida:  gaussiana)
    delta_int(s, sigma) = 
    delta_ext(s, sigma, e, x) = (s, sigma - e)   ignora la entrada
    lambda(s)  = rnd                   (numero aleatorio gaussiano)
    ta(s, sigma) = sigma
"""

import gymnasium as gym
import numpy as np
from pypdevs.DEVS import AtomicDEVS, CoupledDEVS
from pypdevs.infinity import INFINITY

class State:

    def __init__(self, state,sigma):
        self.sigma = sigma
        self.state = state

class MountainCarDevs(AtomicDEVS):

    def __init__(self, name = "MountainCarDEVS", T = 1.0):
        AtomicDEVS.__init__(self, name)

    def int_transition(self, state):

        #El estado queda igual, porque cambia con la delta externa
        #lo que tendría que hacer es asignar el tiempo en infinito, porque queda pasivo

        self.state.sigma = INFINITY
        self.state.state = state

        



