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
        self.criterion = nn.MSELoss().to(self.device)

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
        if train:
          state_tensor = state.unsqueeze(0).to(self.device)
        else:
          state_tensor = self.obs_processing_function(state).unsqueeze(0).to(self.device)

        # Calcular Q-values con policy_net
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)

            # Obtenemos un array de Q-values y seleccionamos la acción greedy
            # con dimensiones 1 x 4
            greedy_action = q_values.argmax(dim=1).item()

        return greedy_action

    def update_weights(self, verbose=False):
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

            stacked_states = torch.stack(states, dim=0).to(
                self.device
            )  # shape: (BATCH, 4, 84, 84)

            print(
                f"Shape of stacked_states: {stacked_states.shape}"
            ) if verbose else None

            # states = torch.tensor(states, dtype=torch.float32, device=self.device)

            # actions = torch.tensor(actions, dtype=torch.int64, device=self.device)

            stacked_actions = torch.tensor(
                actions, dtype=torch.int64, device=self.device
            ).unsqueeze(1)
            print(
                f"Shape of stacked_actions: {stacked_actions.shape}"
            ) if verbose else None

            stacked_rewards = torch.tensor(
                rewards, dtype=torch.float32, device=self.device
            ).unsqueeze(1)
            print(
                f"Shape of stacked_rewards: {stacked_rewards.shape}"
            ) if verbose else None

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

            # 3) Calcular q_current con policy_net(states).gather(...)

            q_s = self.policy_net(stacked_states)
            q_sa = q_s.gather(1, stacked_actions)

            # Validamos que q_sa tenga el shape esperado
            assert q_sa.shape == (self.batch_size, 1), "q_sa no posee el shape esperado"

            # 4) Con torch.no_grad(): calcular max_q_next_state = policy_net(next_states).max(dim=1)[0] * (1 - dones)
            with torch.no_grad():
                q_sn = self.policy_net(stacked_next_states)
                print(f"Shape of q_sn: {q_sn.shape}") if verbose else None
                # Multipliando por (1 - dones) validamos que el episodio haya terminado
                # q_sna = q_sn.max(dim=1)[0] * (1 - stacked_dones)
                q_sna = q_sn.max(1, keepdim=True)[0] * (1 - stacked_dones)

            print(f"Shape of q_sna: {q_sna.shape}") if verbose else None
            # Validamos que max_q_next_state tenga el shape esperado
            assert q_sna.shape == (self.batch_size, 1), (
                "max_q_next_state no posee el shape esperado"
            )

            # 5) Calcular target = rewards + gamma * max_q_next_state
            target = stacked_rewards + self.gamma * q_sna

            # 6) Computar loss MSE entre q_current y target, backprop y optimizer.step()
            # loss = self.criterion(target, q_sa)
            loss = self.criterion(q_sa, target.detach())
            loss.backward()
            self.optimizer.step()

            # 7) Actualizar total_steps
            self.total_steps += 1
            return loss.item()
