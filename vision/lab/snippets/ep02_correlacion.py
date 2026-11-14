import torch
import torch.nn.functional as F
from scipy.signal import convolve2d, correlate2d

torch.manual_seed(0)
# >>>
x = torch.randn(6, 6)
w = torch.tensor([[1., 2., 0.],
                  [0., 0., 0.],
                  [0., 0., -1.]])        # asimétrico a propósito
t = F.conv2d(x[None, None], w[None, None])[0, 0].numpy()
xn, wn = x.numpy(), w.numpy()
corr = correlate2d(xn, wn, mode="valid")
conv = convolve2d(xn, wn, mode="valid")
girada = convolve2d(xn, wn[::-1, ::-1], mode="valid")
print(f"torch vs correlación:  {abs(t - corr).max():.1e}")
print(f"torch vs convolución:  {abs(t - conv).max():.2f}")
print(f"torch vs conv. girada: {abs(t - girada).max():.1e}")
# <<<
