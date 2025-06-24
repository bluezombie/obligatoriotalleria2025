import torch
import torch.nn as nn
import numpy as np
from replay_memory import ReplayMemory, Transition
from abc import ABC, abstractmethod
from tqdm import tqdm
import random
from utils import save_model_checkpoint, load_model_checkpoint


class Agent(ABC):
    def __init__(
        self,
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
    ):
        self.device = device

        # Funcion phi para procesar los estados.
        self.state_processing_function = obs_processing_func

        # Asignarle memoria al agente
        self.memory = ReplayMemory(memory_buffer_size)

        self.env = gym_env

        # Hyperparameters
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma

        self.epsilon_i = epsilon_i
        self.epsilon_f = epsilon_f
        self.epsilon_anneal_steps = epsilon_anneal_steps

        self.episode_block = episode_block

        self.total_steps = 0

    def train(
        self, number_episodes=50_000, max_steps_episode=10_000, max_steps=1_000_000
    ):
        # Cargamos un model checkpoint si existe
        # self.policy_net = load_model_checkpoint(self.policy_net, "./data/")
        # self.policy_net = load_model_checkpoint(self.policy_net, "/content/drive/MyDrive/2025obltalleria/data/")

        rewards = []
        total_steps = 0

        metrics = {"reward": 0.0, "epsilon": self.epsilon_i, "steps": 0}

        pbar = tqdm(range(number_episodes), desc="Entrenando", unit="episode")

        for ep in pbar:
            if total_steps > max_steps:
                break

            # Observar estado inicial como indica el algoritmo
            state, _ = self.env.reset()
            state_phi = self.state_processing_function(state)
            current_episode_reward = 0.0
            current_episode_steps = 0
            done = False

            # Bucle principal de pasos dentro de un episodio
            for _ in range(max_steps):
                # TODO: Seleccionar acción epsilon-greedy usando select_action()

                # Seleccionamos la acción a partir de la implementación propia de
                # la clase abstracta. Luego, cada agente deberá implementar
                # su propia lógica para seleccionar la acción.

                action = self.select_action(state_phi, total_steps, train=True)

                # TODO: Ejecutar action = env.step(action)

                # Ejecutamos la acción.

                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                # TODO: Procesar next_state con state_processing_function

                # Debido a que el ambiente devuelve la imagen sin procesar,
                # debemos prepocesarla para que tenga el formato correcto

                next_state_phi = self.state_processing_function(next_state)

                # TODO: Acumular reward y actualizar total_steps, current_episode_steps

                current_episode_reward += reward
                total_steps += 1
                current_episode_steps += 1

                # TODO: Almacenar transición en replay memory

                # Ingresamos en la memoria de replay la "Transition" definida
                # en la replay memory (tupla).

                self.memory.add(state_phi, action, reward, done, next_state_phi)

                # TODO: Llamar a update_weights() para entrenar modelo

                self.update_weights()

                # TODO: Actualizar state y state_phi al siguiente estado

                state_phi = next_state_phi
                state = next_state

                # TODO: Comprobar condición de done o límite de pasos de episodio y break

                if done or current_episode_steps >= max_steps_episode:
                    break

            # Registro de métricas y progreso
            rewards.append(current_episode_reward)
            metrics["reward"] = np.mean(rewards[-self.episode_block :])
            metrics["epsilon"] = self.compute_epsilon(total_steps)
            metrics["steps"] = total_steps
            pbar.set_postfix(metrics)

            # Guardamos el modelo cada ciertos episodios
            if ep % 1_500 == 0:
                save_model_checkpoint(
                    self.policy_net,
                    f"/content/drive/MyDrive/2025obltalleria_v4/data/dqn_model_{ep}.pth",
                )

        return rewards

    def compute_epsilon(self, steps_so_far):
        """
        Compute el valor de epsilon a partir del número de pasos dados hasta ahora.
        """
        if steps_so_far < self.epsilon_anneal_steps:
            epsilon = self.epsilon_i - (self.epsilon_i - self.epsilon_f) * (
                steps_so_far / self.epsilon_anneal_steps
            )
        else:
            epsilon = self.epsilon_f
        return epsilon

    def play(self, env, episodes=1):
        """
        Modo evaluación: ejecutar episodios sin actualizar la red.
        """
        self.policy_net.eval()  # Aseguramos que el modelo esté en modo evaluación
        for ep in range(episodes):
            state, reset_state = env.reset()
            done = False
            step = 0
            while not done and step < 10_000:
                action = self.select_action(
                    state,
                    step,
                    train=False,
                )
                # Cada 500 pasos imprimimos el estado actual
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                if done:
                    print(f"Episodio {ep + 1} terminado en {step} pasos.")
                    break
                state = next_state
                step += 1
                if step % 500 == 0:
                    print(f"Estado actual: {state}, Paso: {step}")
                    print(
                        f"Episode {ep + 1}, Step {step}: Action {action}, Reward {reward}"
                    )
        self.policy_net.train()  # volvemos al modo entrenamiento

    @abstractmethod
    def select_action(self, state, current_steps, train=True):
        """
        Selecciona una acción a partir del estado actual. Si train=False, se selecciona la acción greedy.
        Si train=True, se selecciona la acción epsilon-greedy.

        Args:
            state: El estado actual del entorno.
            current_steps: El número de pasos actuales. Determina el valor de epsilon.
            train: Si True, se selecciona la acción epsilon-greedy. Si False, se selecciona la acción greedy.
        """
        pass

    @abstractmethod
    def update_weights(self):
        pass
