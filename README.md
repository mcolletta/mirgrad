
# mirgrad


A simple Autograd engine implemented from scratch with numpy for educational purposes.


### Requirements

```bash
pip install numpy
```


### Example usage

Below is a simple example:

```python
import numpy as np
from tensor import Tensor

rng = np.random.default_rng(seed=12345)
i, j, n, m = 7, 5, 3, 2
x1 = rng.random((i, j, n, m), dtype=np.float32)
x2 = rng.random((   1, n, m), dtype=np.float32)
x3 = rng.random((1, 1, n, m), dtype=np.float32)

a = Tensor(x1)
b = Tensor(x2)
c = Tensor(x3)
d = a * b
e = d.exp()
f = c + e

f.backward()

print(f'gradient of a: {a.grad}')
print(f'gradient of b: {b.grad}')
print(f'gradient of c: {c.grad}')
```

### Tests

To run the tests you will have to install [PyTorch](https://pytorch.org/) for verifying the correctness of the calculated gradients.


### References

This work is inspired by the following projects:

[github.com/karpathy/micrograd](https://github.com/karpathy/micrograd)

[github.com/eduardoleao052/Autograd-from-scratch](https://github.com/eduardoleao052/Autograd-from-scratch)

[github.com/ShivamShrirao/simple_autograd_numpy](https://github.com/ShivamShrirao/simple_autograd_numpy)


### License

MIT - see the [LICENSE](https://github.com/mcolletta/mirgrad/blob/main/LICENSE) file for details.
