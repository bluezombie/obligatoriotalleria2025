import random
from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'done', 'next_state'))

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
        
        pass

    def add(self, state, action, reward, done, next_state):
        """
        Agrega una transición a la memoria.
        Si la memoria está llena, sobreescribe la transición más antigua.
        """
        # TODO: crear Transition y agregar o reemplazar en la lista según capacity
        # TODO: actualizar puntero de posición circular
        pass

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
      pass
      
    def __len__(self):
      """
      Devuelve el número actual de transiciones en memoria.
      """
      # TODO: retornar tamaño de la lista de memoria
      pass
    
    def clear(self):
      """
      Elimina todas las transiciones de la memoria.
      """
      # TODO: resetear lista de memoria y puntero de posición
      pass
