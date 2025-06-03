import torch
import torch.nn as nn
import numpy as np
from abstract_agent import Agent
from replay_memory import ReplayMemory, Transition


class DQNAgent(Agent):
    def __init__(
        self,
        env,
        model,
        obs_processing_func,
        memory_buffer_size,
        batch_size,
        learning_rate,
        gamma,
        epsilon_i,
        epsilon_f,
        epsilon_anneal_steps,
        episode_block,
        device,
    ):
        super().__init__(
            env,
            obs_processing_func,
            memory_buffer_size,
            batch_size,
            learning_rate,
            gamma,
            epsilon_i,
            epsilon_f,
            epsilon_anneal_steps,
            episode_block,
            device,
        )
        # Guardar entorno y función de preprocesamiento

        self.env = env
        self.obs_processing_function = obs_processing_func
        self.policy_net = model.to(device)
        self.learning_rate = learning_rate
        self.memory_buffer_size = memory_buffer_size
        self.episode_block = episode_block
        self.total_steps = 0
        self.device = device

        # Inicializar target_net como una copia de policy_net y moverlo al device
        self.target_net = model.to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Inicializar policy_net en device
        self.policy_net.train()  # Asegurarse de que la red está en modo entrenamiento
        self.target_net.eval()  # Asegurarse de que la red objetivo está en modo evaluación

        # Configurar optimizador Adam
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=self.learning_rate
        )

        # Configurar función de pérdida MSE y optimizador Adam
        self.criterion = nn.MSELoss()

        # Crear replay memory de tamaño buffer_size
        self.memory = ReplayMemory(memory_buffer_size)

        # Almacenar batch_size, gamma y parámetros de epsilon-greedy
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon_i = epsilon_i
        self.epsilon_f = epsilon_f
        self.epsilon_anneal_steps = epsilon_anneal_steps
        self.epsilon = epsilon_i

    def select_action(self, state, current_steps, train=True):
        # Calcular epsilon según step
        # Durante entrenamiento: con probabilidad epsilon acción aleatoria
        #                   sino greedy_action

        if train:
            self.epsilon = self.compute_epsilon(current_steps)
            if np.random.rand() < self.epsilon:
                return self.env.action_space.sample()

        # Si no es entrenamiento, obtenemos la acción greedy
        # Procesamos el estado, y lo convertimos a tensor
        state_tensor = self.obs_processing_function(state)

        # Calcular Q-values con policy_net
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)

            # Obtenemos un array de Q-values y seleccionamos la acción greedy
            # con dimensiones 1 x 4
            greedy_action = q_values.argmax(dim=1).item()

        return greedy_action

    def update_weights(self):
        # 1) Comprobar que hay al menos batch_size muestras en memoria
        if len(self.memory) < self.batch_size:
            # 2) Muestrear minibatch y convertir a tensores (states, actions, rewards, dones, next_states)
            transitions = self.memory.sample(self.batch_size)

        # 3) Calcular q_current con policy_net(states).gather(...)
        # 4) Con torch.no_grad(): calcular max_q_next_state = policy_net(next_states).max(dim=1)[0] * (1 - dones)
        # 5) Calcular target = rewards + gamma * max_q_next_state
        # 6) Computar loss MSE entre q_current y target, backprop y optimizer.step()
        pass
