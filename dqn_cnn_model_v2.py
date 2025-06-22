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


class DQN_CNN_Model_v2(nn.Module):
    def __init__(self, obs_shape, n_actions, conv_config=[16, 32]):
        super(DQN_CNN_Model_v2, self).__init__()

        self.conv_layers = nn.ModuleList()
        in_channels = obs_shape[0]

        # Definimos las dimensiones para poder
        # obtener el valor final de la salida.
        H_out, W_out = obs_shape[1], obs_shape[2]
        print(f"Input shape: {obs_shape}")

        for out_channels in conv_config:
            layer = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            self.conv_layers.append(layer)
            # Calculamos la salida de la capa convolucional
            H_out, W_out = conv2d_output_shape(
                (H_out, W_out),
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
            )
            in_channels = out_channels  # update for next layer

        self.fc1 = nn.Linear(in_channels * H_out * W_out, 256)
        self.output = nn.Linear(256, n_actions)

    def forward(self, obs):
        result = obs
        for conv in self.conv_layers:
            result = F.relu(conv(result))

        # Aplanamos la salida de las capas convolucionales
        result = result.view(result.size(0), -1)

        # Aplicamos las capas lineales
        result = F.relu(self.fc1(result))
        result = self.output(result)

        return result
