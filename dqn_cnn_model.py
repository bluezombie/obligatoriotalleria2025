import torch.nn as nn
import torch.nn.functional as F


def conv2d_output_shape(
    input_size: tuple[int, int],
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
) -> tuple[int, int]:
    """
    Calcula (H_out, W_out) para una capa Conv2d con:
      - input_size: (H_in, W_in)
      - kernel_size, stride, padding, dilation: int o tupla (altura, ancho)
    Basado en:
      H_out = floor((H_in + 2*pad_h - dil_h*(ker_h−1) - 1) / str_h + 1)
      W_out = floor((W_in + 2*pad_w - dil_w*(ker_w−1) - 1) / str_w + 1)
    Fuente: Shape section en torch.nn.Conv2d :contentReference[oaicite:0]{index=0}
    """

    # Unifica todos los parámetros a tuplas (h, w)
    def to_tuple(x):
        return (x, x) if isinstance(x, int) else x

    H_in, W_in = input_size
    ker_h, ker_w = to_tuple(kernel_size)
    str_h, str_w = to_tuple(stride)
    pad_h, pad_w = to_tuple(padding)
    dil_h, dil_w = to_tuple(dilation)

    H_out = (H_in + 2 * pad_h - dil_h * (ker_h - 1) - 1) // str_h + 1
    W_out = (W_in + 2 * pad_w - dil_w * (ker_w - 1) - 1) // str_w + 1

    return H_out, W_out


class DQN_CNN_Model(nn.Module):
    def __init__(self, obs_shape, n_actions):
        super(DQN_CNN_Model, self).__init__()
        # TODO: definir capas convolucionales basadas en obs_shape
        # TODO: definir capas lineales basadas en n_actions

        # Obtenemos la cantidad de canales
        channels = obs_shape[0]

        # Recordemos que "channels" representa la cantidad de estados
        # que componen a una "transición" (es decir, un conjunto
        # de imágenes).
        # Por lo tanto, la cantidad de canales de entrada es igual
        # a la cantidad de estados.

        # Definimos la primera capa convolucional de acuerdo
        # al paper de DQN:
        #  - kernel_size=8
        #  - stride=4
        #  - filters=16

        self.conv1 = nn.Conv2d(channels, 16, kernel_size=8, stride=4)

        # Definimos la segundo capa convolucional:
        #  - kernel_size=8
        #  - stride=4
        #  - filters=16

        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)

        # Calculemos a continuación el tamaño de la última capa
        # oculta de la red convolucional para poder obtener
        # la cantidad de entradas para la capa lineal.

        # La salida de la primera capa convolucional es:
        # W_out = [ (W_in - W_k + 2*pad) / stride ] + 1
        # Por tanto:
        # W_out = [ (84 - 8 + 2*0) / 4 ] + 1 = 20

        # La salida de la segunda capa convolucional es:
        # W_out = [ (20 - 4 + 2*0) / 2 ] + 1 = 9

        # Por lo tanto, la cantidad de entradas para la capa
        # lineal es:
        # 32 * 9 * 9 = 2592
        self.fc1 = nn.Linear(32 * 9 * 9, 256)

        # Luego, la salida queda definda por la cantidad
        # de acciones (dependiendo del ambiente)
        self.output = nn.Linear(256, n_actions)

    def forward(self, obs):
        # TODO: 1) aplicar convoluciones y activaciones
        #       2) aplanar la salida
        #       3) aplicar capas lineales
        #       4) devolver tensor de Q-values de tamaño (batch, n_actions)

        # 1.a Aplicamos la primera capa convolucional
        # con su respectiva función de activación.
        result = F.relu(self.conv1(obs))

        # 1.b Aplicamos la segunda capa convolucional
        result = F.relu(self.conv2(result))

        # 2 Aplanamos la salida
        result = result.view(result.size(0), -1)

        # 3 Aplicamos las capas lineales
        result = F.relu(self.fc1(result))
        result = self.output(result)

        # 4 Devolvemos el tensor de Q-values

        return result
