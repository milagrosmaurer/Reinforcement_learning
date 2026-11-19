from mountainCarModularizado.acompladoMC import main
from comparacion_gym.aprendizaje_gym import train

if __name__ == '__main__':
    train(10000, 'corrida_gym_17_11', seed= 22, guardar = True)

    main('corrida_devs_17_11','MC_17_11',2000000.0, seed = 22, entrenamiento=True)
