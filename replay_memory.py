import random
from collections import namedtuple

Transition = namedtuple(
    "Transition", ("state", "action", "reward", "done", "next_state")
)

# Ejemplo uso
# nueva_tupla = Transition(state, action, reward, done, next_state)


class ReplayMemory:
    def __init__(self, capacity):
        """
        Inicializa la memoria de repetición con capacidad fija.
        Params:
         - capacity (int): número máximo de transiciones a almacenar.
        """
        # TODO: almacenar capacity, inicializar lista de memoria y puntero de posición
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def add(self, state, action, reward, done, next_state):
        """s
        Agrega una transición a la memoria.
        Si la memoria está llena, sobreescribe la transición más antigua.
        """
        # TODO: crear Transition y agregar o reemplazar en la lista según capacity
        # TODO: actualizar puntero de posición circular\
        transition = Transition(state, action, reward, done, next_state)
        if len(self.memory) < self.capacity:
            # Si la memoria NO está llena, agregamos al final
            # de la lista.
            self.memory.append(transition)
        else:
            # Si la memoria está llena, reemplazamos la transición
            # de acuerdo a la posición de la memoria circular.
            self.memory[self.position] = transition
        # Por último, actualizamos la posición de la memoria circular
        # en base al "resto" de la división entre la posición más uno,
        # y la capacidad de la memoria. Esto quiere decir que
        # cuando la posición alcance la capacidad,
        # el contador volverá a cero.
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Devuelve un batch aleatorio de transiciones.
        Params:
         - batch_size (int): número de transiciones a muestrear.
        Returns:
         - lista de Transition de longitud batch_size.
        """
        # TODO: verificar que batch_size <= len(self)
        # TODO: retornar una muestra aleatoria de self.memory
        if batch_size > len(self.memory):
            raise ValueError(
                "cantidad (batch_size) mayor a la cantidad de transiciones en memoria "
            )
        sample = random.sample(self.memory, batch_size)
        return sample

    def __len__(self):
        """
        Devuelve el número actual de transiciones en memoria.
        """
        # TODO: retornar tamaño de la lista de memoria
        return len(self.memory)

    def clear(self):
        """
        Elimina todas las transiciones de la memoria.
        """
        # TODO: resetear lista de memoria y puntero de posición
        self.memory = []
        self.position = 0
