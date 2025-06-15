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
        if len(self.memory) > self.batch_size:
            # Debemos resetear los gradientes del optimizador
            self.optimizer.zero_grad()

            # 2) Muestrear minibatch y convertir a tensores (states, actions, rewards, dones, next_states)
            transitions = self.memory.sample(
                self.batch_size
            )  # Sampleamos un batch de transiciones

            states, actions, rewards, dones, next_states = zip(
                *transitions
            )  # A partir de las tuplas,
            # deshacemos el zip para obtener las variables de interés

            # Convertimos las listas en tensores y los movemos al dispositivo
            states = torch.tensor(states, dtype=torch.float32, device=self.device)
            actions = torch.tensor(actions, dtype=torch.int64, device=self.device)
            rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
            dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
            next_states = torch.tensor(
                next_states, dtype=torch.float32, device=self.device
            )

            # Realicemos algunos controles de los datos, validando que el shape
            # de los tensores son correctos

            assert states.shape == (self.batch_size, 4, 84, 84), (
                "Los estados no poseen el shape esperado"
            )
            assert next_states.shape == (self.batch_size, 4, 84, 84), (
                "Los estados próximos no poseen el shape esperado"
            )

            assert actions.shape == (self.batch_size, 1), (
                "Las acciones no poseen el shape esperado"
            )
            assert rewards.shape == (self.batch_size, 1), (
                "Las recompensas no poseen el shape esperado"
            )
            assert dones.shape == (self.batch_size, 1), (
                "Los dones no poseen el shape esperado"
            )

            # 3) Calcular q_current con policy_net(states).gather(...)

            q_s = self.policy_net(states)
            q_sa = q_s.gather(1, actions)

            # Validamos que q_sa tenga el shape esperado
            assert q_sa.shape == (self.batch_size, 1), "q_sa no posee el shape esperado"

            # 4) Con torch.no_grad(): calcular max_q_next_state = policy_net(next_states).max(dim=1)[0] * (1 - dones)
            with torch.no_grad():
                q_sn = self.policy_net(next_states)
                # Multipliando por (1 - dones) validamos que el episodio haya terminado
                q_sna = q_sn.max(dim=1)[0] * (1 - dones)

            # Validamos que max_q_next_state tenga el shape esperado
            assert q_sna.shape == (self.batch_size, 1), (
                "max_q_next_state no posee el shape esperado"
            )

            # 5) Calcular target = rewards + gamma * max_q_next_state
            target = rewards + self.gamma * q_sna

            # 6) Computar loss MSE entre q_current y target, backprop y optimizer.step()
            loss = self.criterion(target, q_sa)
            loss.backward()
            self.optimizer.step()

            # 7) Actualizar total_steps
            self.total_steps += 1
            return loss.item()
