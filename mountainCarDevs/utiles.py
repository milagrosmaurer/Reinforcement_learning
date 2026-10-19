import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import numpy as np
import torch

FORCE = 0.001
GRAVITY = 0.0025

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 64)
        self.layer2 = nn.Linear(64, n_actions)
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        return self.layer2(x)

@dataclass(slots=True)
class TransicionDatos:
    state: list | None =  None
    action: int = None
    next_state: list | None = None
    reward: int = None
    terminated: bool = None
    truncated: bool = None


@dataclass
class HiperParametros:
    batch_size: int = 128
    memory_capacity: int = 10000
    eps_end: float = 0.001
    eps_start: float = 0.99
    eps_decay: int = 4000
    gamma: float = 0.99
    lr: float = 3e-4

def chequeo(estado: list, action: int) -> list[float]:
    estado = estado[0]
    pos = estado[0]
    vel = estado[1]

    vel_nue = vel + (action - 1)*FORCE - np.cos(3*pos)*GRAVITY
    pos_nue = pos + vel_nue

    vel = np.clip(vel_nue,-0.07, 0.07)
    pos = np.clip(pos_nue, -1.2, 0.6)
    return [pos, vel]




    