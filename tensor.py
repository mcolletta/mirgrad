import numpy as np
from itertools import zip_longest


class Tensor:
    def __init__(self, data, prev=(), grad_fn=None, *args, **kwargs):
        self.data = np.asarray(data)
        self.prev = prev
        self.grad = 0
        self.grad_fn = grad_fn if grad_fn else lambda x: None

    def topological_sort(self, dag):
        tpl = []
        deps = []
        visited = set()
        to_visit = [dag]
        while to_visit:
            node = to_visit.pop()
            if node not in visited:
                visited.add(node)
                to_visit.extend(node.prev)
                while deps and node not in deps[-1].prev:
                    tpl.append(deps.pop())
                deps.append(node)
        tpl = deps + tpl[::-1]
        return tpl

    def backward(self, gradient=None):
        if not gradient:
            gradient = np.ones_like(self.data)
        self.grad = gradient
        flat_dag = self.topological_sort(self)        
        for node in flat_dag:
            if node.prev:
                node.grad_fn(node.prev, node.grad)

    def zero_grad(self):
        self.grad = 0
        if self.prev:
            for t in self.prev:
                t.zero_grad()

    def __repr__(self):
        r = repr(self.data)
        return r[:10].replace('array','tensor') + r[10:]

    def __add__(self, other):
        return Add.apply(self, other)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __mul__(self, other):
        return Mul.apply(self, other)

    def __rmul__(self, other):
        return self * other

    def __matmul__(self, other):
        return MatMul.apply(self, other)

    def __pow__(self, other):
        return  Pow.apply(self, other)

    def sin(self):
        return Sin.apply(self)

    def cos(self):
        return Cos.apply(self)

    def exp(self):
        return Exp.apply(self)

    def log(self):
        return Log.apply(self)

    def relu(self):
        return Relu.apply(self)

    def sigmoid(self):
        return Sigmoid.apply(self)

    def __neg__(self):
        return self * -1

    def __truediv__(self, other):
        return self * (other**-1)

    def __rtruediv__(self, other):
        return other * self**-1

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype


def scale_grad(grad, t):
    grad_dim = len(grad.shape)
    t_dim = len(t.shape)
    new_grad = grad
    for _ in range(grad_dim - t_dim):
        new_grad = new_grad.sum(axis=0)
    bdims = []
    for n, dim in enumerate(t.shape):
        if dim == 1:
            bdims.append(n)
    if bdims:
        new_grad = new_grad.sum(axis=tuple(bdims), keepdims=True)
    return new_grad


class Add:

    @staticmethod
    def apply(x, y):
        y = y if isinstance(y, Tensor) else Tensor(y)
        return Tensor(x.data + y.data, (x, y), grad_fn=Add.compute_grad)

    @staticmethod
    def compute_grad(ctx, gradient):
        x, y = ctx
        # take care of a possible broadcast from numpy in the apply method
        x.grad += scale_grad(gradient, x)
        y.grad += scale_grad(gradient, y)


class Mul:

    @staticmethod
    def apply(x, y):
        y = y if isinstance(y, Tensor) else Tensor(y)
        return Tensor(x.data * y.data, (x, y), grad_fn=Mul.compute_grad)

    @staticmethod
    def compute_grad(ctx, gradient):
        x, y = ctx
        dx = y.data * gradient
        x.grad += scale_grad(dx, x)
        dy = x.data * gradient
        y.grad += scale_grad(dy, y)


class MatMul:

    @staticmethod
    def apply(x, y):
        y = y if isinstance(y, Tensor) else Tensor(y)
        return Tensor(x.data @ y.data, (x, y), grad_fn=MatMul.compute_grad)

    @staticmethod
    def compute_grad(ctx, gradient):
        x, y = ctx
        # transpose only the two dimensions of matrix     
        dx = gradient @ np.moveaxis(y.data, -1, -2)
        # broadcast for the dimensions not part of 2d matrix mult
        x.grad += scale_grad(dx, x)
        dy = np.moveaxis(x.data, -1, -2) @ gradient
        y.grad += scale_grad(dy, y)


class Pow:

    @staticmethod
    def apply(x, n):
        n = n if isinstance(n, Tensor) else Tensor(n)
        return Tensor(x.data ** n.data, (x, n), grad_fn=Pow.compute_grad)

    @staticmethod
    def compute_grad(ctx, gradient):
        x, n = ctx
        x.grad += gradient * (n.data * (x.data ** (n.data - 1)))


class Sin:

    @staticmethod
    def apply(x):
        return Tensor(np.sin(x.data), (x, ), grad_fn=Sin.compute_grad)

    @staticmethod
    def compute_grad(ctx, gradient):
        x, = ctx
        x.grad += gradient * np.cos(x.data)


class Cos:

    @staticmethod
    def apply(x):
        return Tensor(np.cos(x.data), (x, ), grad_fn=Cos.compute_grad)

    @staticmethod
    def compute_grad(ctx, gradient):
        x, = ctx
        x.grad += gradient * -np.sin(x.data)


class Exp:

    @staticmethod
    def apply(x):
        return Tensor(np.exp(x.data), (x, ), grad_fn=Exp.compute_grad)

    @staticmethod
    def compute_grad(ctx, gradient):
        x, = ctx
        x.grad += gradient * np.exp(x.data)


class Log:

    @staticmethod
    def apply(x):
        return Tensor(np.log(x.data), (x, ), grad_fn=Log.compute_grad)

    @staticmethod
    def compute_grad(ctx, gradient):
        x, = ctx
        x.grad += gradient * (1. / x.data)


class Relu:

    @staticmethod
    def apply(x):
        return Tensor(x.data*(x.data > 0), (x, ), grad_fn=Relu.compute_grad)


    @staticmethod
    def compute_grad(ctx, gradient):
        x, = ctx
        x.grad += gradient * (x.data > 0)


class Sigmoid:

    @staticmethod
    def apply(x):
        return Tensor(1.0 / (1 + np.exp(-x.data)), (x, ), grad_fn=Sigmoid.compute_grad)

    @staticmethod
    def compute_grad(ctx, gradient):
        x, = ctx
        sigma = 1.0 / (1 + np.exp(-x.data))
        x.grad += gradient * (sigma * (1 - sigma))


def check_broadcast(x, y, skip=0):
    x_bdim = []
    y_bdim = []
    # x.shape = (7, 1, 3, 2)
    # y.shape =    (5, 2, 9)
    # [(i,j) for (n,(i,j)) in enumerate(zip_longest(reversed(x.shape), reversed(y.shape)))]
    # [(2, 9), (3, 2), (1, 5), (7, None)]
    # [(n,(i,j)) for (n,(i,j)) in enumerate( reversed( list( zip_longest(reversed(x.shape), reversed(y.shape)) ) ) )]
    # [(0, (7, None)), (1, (1, 5)), (2, (3, 2)), (3, (2, 9))]
    num = max(len(x.shape), len(y.shape))  # 4
    stop = num - skip  # 4 - 2 = 2
    for n,(i,j) in enumerate(reversed(list(zip_longest(reversed(x.shape), reversed(y.shape))))):
        if n >= stop or (i is None) or (j is None) or i == j:
            continue
        elif i < j:
            x_bdim.append(n)
        else:
            y_bdim.append(n)
    return x_bdim, y_bdim
