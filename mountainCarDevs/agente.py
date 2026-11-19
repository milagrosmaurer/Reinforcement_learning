import numpy as np
from pypdevs.DEVS import AtomicDEVS
from pypdevs.infinity import INFINITY
import torch
from .utiles import DQN
import json

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

class AgenteState:
    def __init__(self, action = None, sigma = INFINITY):
        self.action = action
        self.sigma = sigma

class Agente(AtomicDEVS):
    def __init__(self, nombre_archivo,seed, epsilon = None, pesos = None,name = "agente"):
        AtomicDEVS.__init__(self, name=name)
        self.state = AgenteState()
        torch.manual_seed(seed=seed)
        self.rng = np.random.default_rng(seed)
        self.nombre_archivo = nombre_archivo
        self.policy_net = DQN(2,3).to(device = device)
        self.policy_net.eval()
        self.PESOS = self.addInPort("PESOS")
        self.TRANSICION = self.addInPort("TRANSICION")
        self.ACCION = self.addOutPort("ACCION")
        self.epsilon = 1.0 if not epsilon else epsilon
        self.pasos = 0
        self.eps_end = 0.01
        self.eps_start = 0.99
        self.eps_decay = 4000
        self.episodios = 1
        self.epsilons = []
        if pesos:
            cargar_pesos = torch.load(pesos, weights_only=True)
            self.policy_net.load_state_dict(cargar_pesos)
        

    def intTransition(self):
        self.state.sigma = INFINITY
        return self.state

    def extTransition(self, inputs):
        
        pesos = inputs.get(self.PESOS)
        transicion = inputs.get(self.TRANSICION)
        
        if pesos is not None:
            self.policy_net.load_state_dict(pesos)
            self.state.sigma = INFINITY

        if transicion is not None:
            estado_actual, reward, terminated, truncated = transicion
            self.pasos += 1
            if terminated or truncated:
                self.pasos = 0
                self.episodios += 1
                self.state.sigma = INFINITY
                self.epsilons.append(self.epsilon)
                if self.episodios % 500 == 0:
                    with open(f"resultados/semillas/resultados_eps_{self.nombre_archivo}.json", "w") as archivo:
                        json.dump({
                            "eps": self.epsilons
                        }, archivo)
            else:
                self.state.sigma = 0.0

            self.epsilon = max(
                self.eps_end,
                self.eps_start - (self.eps_start - self.eps_end) * (self.episodios / self.eps_decay)
            )

            if self.rng.random() < self.epsilon:
                
                self.state.action = torch.tensor([[self.rng.integers(0,3)]], dtype = torch.long, device=device)
            else:
                estado_actual_tensor = torch.tensor(estado_actual, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    self.state.action = self.policy_net(estado_actual_tensor).max(1).indices.view(1,1)

        return self.state

    def outputFnc(self):
        return {self.ACCION : self.state.action.item()}

    def timeAdvance(self):
        return self.state.sigma

    

        

                

