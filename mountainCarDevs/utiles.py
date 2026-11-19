import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import numpy as np
import torch
import json
import matplotlib.pyplot as plt

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
    #estado = estado[0]
    pos = estado[0]
    vel = estado[1]

    vel_nue = vel + (action - 1)*FORCE - np.cos(3*pos)*GRAVITY
    pos_nue = pos + vel_nue
    print(f'La nueva velocidad calculada es {vel_nue}')
    

    #vel = np.clip(vel_nue,-0.07, 0.07)
    #pos = np.clip(pos_nue, -1.2, 0.6)
    return [pos_nue, vel_nue]


def graficar_recompensas(archivo_recompensas: str, episodios:int =  10, window: int = 100):
    with open(archivo_recompensas) as f:
        recompensas = json.load(f)

    recompensas_promedio = np.zeros(len(recompensas))
    for i in range(len(recompensas)):
        recompensas_promedio[i] = np.mean(recompensas[max(0,i - window):i+1])

    plt.figure(figsize=(10, 5))

    plt.plot(recompensas_promedio, color='royalblue', linewidth=2, label='Recompensa promedio')

    plt.xlabel('Episodios', fontsize=12)
    plt.ylabel(f'Promedio de recompensas (ventana = {window})', fontsize=12)
    plt.title('Evolución de las recompensas durante el entrenamiento', fontsize=14)

    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    graficar_recompensas(r'C:\Users\Milagritos\Desktop\Devs\datos.json')


    