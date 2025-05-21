import torch
import torch.nn as nn
import numpy as np
from abstract_agent import Agent

class DQNAgent(Agent):
    def __init__(self, env, model, obs_processing_func, memory_buffer_size, batch_size, learning_rate, gamma, epsilon_i, epsilon_f, epsilon_anneal_steps, episode_block, device):
        super().__init__(env, obs_processing_func, memory_buffer_size, batch_size, learning_rate, gamma, epsilon_i, epsilon_f, epsilon_anneal_steps, episode_block, device)
        # Guardar entorno y función de preprocesamiento
        # Inicializar policy_net en device
        # Configurar función de pérdida MSE y optimizador Adam
        # Crear replay memory de tamaño buffer_size
        # Almacenar batch_size, gamma y parámetros de epsilon-greedy
        pass
        
    def select_action(self, state, current_steps, train=True):
      # Calcular epsilon según step
      # Durante entrenamiento: con probabilidad epsilon acción aleatoria
      #                   sino greedy_action
      # Durante evaluación: usar greedy_action (o pequeña epsilon fija)
      pass

    def update_weights(self):
      # 1) Comprobar que hay al menos batch_size muestras en memoria
      # 2) Muestrear minibatch y convertir a tensores (states, actions, rewards, dones, next_states)
      # 3) Calcular q_current con policy_net(states).gather(...)
      # 4) Con torch.no_grad(): calcular max_q_next_state = policy_net(next_states).max(dim=1)[0] * (1 - dones)
      # 5) Calcular target = rewards + gamma * max_q_next_state
      # 6) Computar loss MSE entre q_current y target, backprop y optimizer.step()
      pass