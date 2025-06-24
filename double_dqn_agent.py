import torch
import torch.nn as nn
import torch.nn.functional as F
from replay_memory import ReplayMemory, Transition
import numpy as np
from abstract_agent import Agent
import random


class DoubleDQNAgent(Agent):
    def __init__(
        self,
        gym_env,
        model_a,
        model_b,
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
        sync_target=1000,
    ):
        super().__init__(
            gym_env,
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

        self.env = gym_env
        self.obs_processing_function = obs_processing_func

        # Inicializar online_net (model_a) y target_net (model_b) en device

        self.online_net = model_a.to(device)
        self.target_net = model_b.to(device)

        # Seteamos la target_net en modo evaluación
        self.target_net.eval()

        # Configurar función de pérdida MSE y optimizador Adam para online_net
        self.criterion = nn.MSELoss().to(self.device)

        self.optimizer = torch.optim.Adam(
            self.online_net.parameters(), lr=self.learning_rate
        )
        # Crear replay memory de tamaño buffer_size
        self.memory_buffer_size = memory_buffer_size
        self.memory = ReplayMemory(memory_buffer_size)

        # Almacenar batch_size, gamma, parámetros de epsilon y sync_target
        self.sync_target = sync_target
        self.sync_counter = sync_target

        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon_i = epsilon_i
        self.epsilon_f = epsilon_f
        self.epsilon_anneal_steps = epsilon_anneal_steps
        self.epsilon = epsilon_i
        self.learning_rate = learning_rate
        self.episode_block = episode_block
        self.total_steps = 0
        self.device = device

    def select_action(self, state, current_steps, train=True):
        # Calcular epsilon según step
        # Durante entrenamiento: con probabilidad epsilon acción aleatoria
        #                   sino greedy_action

        if train and np.random.rand() < self.compute_epsilon(current_steps):
            return self.env.action_space.sample()

        # Si no estoy entrenando o no quiero explorar,
        # tomamos el estado pasado como parámetro
        # y seleccionamos la acción greedy.
        if train:
            state_tensor = state.unsqueeze(0).to(self.device)
        else:
            state_tensor = (
                self.obs_processing_function(state).unsqueeze(0).to(self.device)
            )

        # Calcular Q-values con policy_net
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)

            # Obtenemos un array de Q-values y seleccionamos la acción greedy
            # con dimensiones 1 x 4
            greedy_action = q_values.argmax(dim=1).item()

        return greedy_action

    def update_weights(self):
        # 1) Verificar que haya al menos batch_size transiciones en memoria
        if len(self.memory) > self.batch_size:
            # 2) Muestrear minibatch y convertir estados, acciones, recompensas, dones y next_states a tensores
            transitions = self.memory.sample(
                self.batch_size
            )  # Sampleamos un batch de transiciones

            # A partir de las tuplas,
            # deshacemos el zip para obtener las variables de interés
            states, actions, rewards, dones, next_states = zip(*transitions)

            stacked_states = torch.stack(states, dim=0).to(self.device)

            stacked_actions = torch.tensor(
                actions, dtype=torch.int64, device=self.device
            ).unsqueeze(1)

            stacked_rewards = torch.tensor(
                rewards, dtype=torch.float32, device=self.device
            ).unsqueeze(1)

            stacked_dones = torch.tensor(
                dones, dtype=torch.float32, device=self.device
            ).unsqueeze(1)

            stacked_next_states = torch.stack(next_states, dim=0).to(self.device)

            # Realicemos algunos controles de los datos, validando que el shape
            # de los tensores son correctos

            assert stacked_states.shape == (self.batch_size, 4, 84, 84), (
                "Los estados no poseen el shape esperado"
            )

            assert stacked_next_states.shape == (self.batch_size, 4, 84, 84), (
                "Los estados próximos no poseen el shape esperado"
            )

            assert stacked_actions.shape == (self.batch_size, 1), (
                "Las acciones no poseen el shape esperado"
            )

            assert stacked_rewards.shape == (self.batch_size, 1), (
                "Las recompensas no poseen el shape esperado"
            )

            assert stacked_dones.shape == (self.batch_size, 1), (
                "Los dones no poseen el shape esperado"
            )

            # 3) Calcular q_current: online_net(states).gather(…)
            q_s = self.online_net(stacked_states)
            q_current = q_s.gather(1, stacked_actions)

            # Validamos que q_current tenga el shape esperado
            assert q_current.shape == (self.batch_size, 1), (
                "q_sa no posee el shape esperado"
            )

            # 4) Calcular target Double DQN:

            #    a) best_actions = online_net(next_states).argmax(…)
            best_actions = self.online_net(stacked_next_states).argmax(1, keepdim=True)
            #    b) q_next = target_net(next_states).gather(… best_actions)
            q_next = self.target_net(stacked_next_states).gather(1, best_actions)
            #    c) target_q = rewards + gamma * q_next * (1 - dones)
            target_q = stacked_rewards + self.gamma * q_next * (1 - stacked_dones)
            # 5) Computar loss MSE entre q_current y target_q, backprop y optimizer.step()

            self.optimizer.zero_grad()  # Limpiar gradientes del optimizador

            loss = self.criterion(q_current, target_q.detach())
            loss.backward()  # Calcular gradientes
            self.optimizer.step()  # Actualizar pesos del modelo online

            # 6) Decrementar contador y si llega a 0 copiar online_net → target_net
            self.sync_counter -= 1
            if self.sync_counter <= 0:
                self.target_net.load_state_dict(self.online_net.state_dict())
                self.sync_counter = self.sync_target
