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

    H_out = (H_in + 2*pad_h - dil_h*(ker_h - 1) - 1) // str_h + 1
    W_out = (W_in + 2*pad_w - dil_w*(ker_w - 1) - 1) // str_w + 1

    return H_out, W_out


class DQN_CNN_Model(nn.Module):
    def __init__(self,  obs_shape, n_actions):
        super(DQN_CNN_Model, self).__init__()
        # TODO: definir capas convolucionales basadas en obs_shape
        # TODO: definir capas lineales basadas en n_actions
        pass
        

    def forward(self, obs):
        # TODO: 1) aplicar convoluciones y activaciones
        #       2) aplanar la salida
        #       3) aplicar capas lineales
        #       4) devolver tensor de Q-values de tamaño (batch, n_actions)
        pass