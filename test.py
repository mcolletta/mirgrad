import numpy as np
import torch
import torch.nn.functional as F
import random
from tensor import Tensor


torch.manual_seed(12345)
random.seed(12345)


def test_add():
    print("test_add")

    pt_a = torch.rand((7, 5, 2, 3), requires_grad=True)
    pt_b = torch.rand((   1, 2, 3), requires_grad=True)
    pt_c = pt_a + pt_b
    pt_c.backward(torch.ones_like(pt_c))

    mir_a = Tensor(pt_a.detach().cpu().numpy())
    mir_b = Tensor(pt_b.detach().cpu().numpy())
    mir_c = mir_a + mir_b
    mir_c.backward()

    assert np.allclose(pt_a.grad.detach().cpu().numpy(), mir_a.grad), "Not equal a grad"
    assert np.allclose(pt_b.grad.detach().cpu().numpy(), mir_b.grad), "Not equal b grad"


def test_mul():
    print("test_mul")

    pt_a = torch.rand((7, 1, 3, 2), requires_grad=True)
    pt_b = torch.rand((   5, 3, 2), requires_grad=True)
    pt_c = pt_a * pt_b
    pt_c.backward(torch.ones_like(pt_c))

    mir_a = Tensor(pt_a.detach().cpu().numpy())
    mir_b = Tensor(pt_b.detach().cpu().numpy())
    mir_c = mir_a * mir_b
    mir_c.backward()

    assert np.allclose(pt_a.grad.detach().cpu().numpy(), mir_a.grad), "Not equal a grad"
    assert np.allclose(pt_b.grad.detach().cpu().numpy(), mir_b.grad), "Not equal b grad"


def test_matmul():
    print("test_matmul")

    pt_a = torch.rand((7, 1, 3, 2), requires_grad=True)
    pt_b = torch.rand((   5, 2, 9), requires_grad=True)
    pt_c = pt_a @ pt_b
    pt_c.backward(torch.ones_like(pt_c))

    mir_a = Tensor(pt_a.detach().cpu().numpy())
    mir_b = Tensor(pt_b.detach().cpu().numpy())
    mir_c = mir_a @ mir_b
    mir_c.backward()

    assert np.allclose(pt_a.grad.detach().cpu().numpy(), mir_a.grad), "Not equal a grad"
    assert np.allclose(pt_b.grad.detach().cpu().numpy(), mir_b.grad), "Not equal b grad"


def test_pow():
    print("test_pow")

    pt_x = torch.rand((3, 1, 2, 3), requires_grad=True)
    pt_y= pt_x ** 3
    pt_y.backward(torch.ones_like(pt_y))
    mir_x = Tensor(pt_x.detach().cpu().numpy())
    mir_y = mir_x ** 3
    mir_y.backward()
    assert np.allclose(pt_x.grad.detach().cpu().numpy(), mir_x.grad), "Not equal x grad"


def test_exp():
    print("test_exp")

    pt_x = torch.rand((3, 1, 2, 3), requires_grad=True)
    pt_y= pt_x.exp()
    pt_y.backward(torch.ones_like(pt_y))
    mir_x = Tensor(pt_x.detach().cpu().numpy())
    mir_y = mir_x.exp()
    mir_y.backward()
    assert np.allclose(pt_x.grad.detach().cpu().numpy(), mir_x.grad), "Not equal x grad"


def test_log():
    print("test_log")

    pt_x = torch.rand((3, 1, 2, 3), requires_grad=True)
    pt_y= pt_x.log()
    pt_y.backward(torch.ones_like(pt_y))
    mir_x = Tensor(pt_x.detach().cpu().numpy())
    mir_y = mir_x.log()
    mir_y.backward()
    assert np.allclose(pt_x.grad.detach().cpu().numpy(), mir_x.grad), "Not equal x grad"


def test_sin():
    print("test_sin")

    pt_x = torch.rand((3, 1, 2, 3), requires_grad=True)
    pt_y= pt_x.sin()
    pt_y.backward(torch.ones_like(pt_y))
    mir_x = Tensor(pt_x.detach().cpu().numpy())
    mir_y = mir_x.sin()
    mir_y.backward()
    assert np.allclose(pt_x.grad.detach().cpu().numpy(), mir_x.grad), "Not equal x grad"


def test_cos():
    print("test_sin")

    pt_x = torch.rand((3, 1, 2, 3), requires_grad=True)
    pt_y= pt_x.cos()
    pt_y.backward(torch.ones_like(pt_y))
    mir_x = Tensor(pt_x.detach().cpu().numpy())
    mir_y = mir_x.cos()
    mir_y.backward()
    assert np.allclose(pt_x.grad.detach().cpu().numpy(), mir_x.grad), "Not equal x grad"


def test_sigmoid():
    print("test_sigmoid")

    pt_x = torch.rand((3, 1, 2, 3), requires_grad=True)
    pt_y= pt_x.sigmoid()
    pt_y.backward(torch.ones_like(pt_y))
    mir_x = Tensor(pt_x.detach().cpu().numpy())
    mir_y = mir_x.sigmoid()
    mir_y.backward()
    assert np.allclose(pt_x.grad.detach().cpu().numpy(), mir_x.grad), "Not equal x grad"


def test_relu():
    print("test_relu")

    pt_x = torch.rand((3, 1, 2, 3), requires_grad=True)
    pt_y= F.relu(pt_x)
    pt_y.backward(torch.ones_like(pt_y))
    mir_x = Tensor(pt_x.detach().cpu().numpy())
    mir_y = mir_x.relu()
    mir_y.backward()
    assert np.allclose(pt_x.grad.detach().cpu().numpy(), mir_x.grad), "Not equal x grad"


def test_operation1():
    print("test_operation1")
    i, j, n, m = 7, 5, 3, 2

    pt_a = torch.rand((i, j, n, m), requires_grad=True)
    pt_b = torch.rand((1, n, m), requires_grad=True)
    pt_c = torch.rand((1, 1, n, m), requires_grad=True)
    pt_d = pt_a * pt_b
    pt_e = pt_d.exp()
    pt_f = pt_c + pt_e
    pt_f.backward(torch.ones_like(pt_f))

    mir_a = Tensor(pt_a.detach().cpu().numpy())
    mir_b = Tensor(pt_b.detach().cpu().numpy())
    mir_c = Tensor(pt_c.detach().cpu().numpy())
    mir_d = mir_a * mir_b
    mir_e = mir_d.exp()
    mir_f = mir_c + mir_e
    mir_f.backward()

    assert np.allclose(pt_a.grad.detach().cpu().numpy(), mir_a.grad), "Not equal a grad"
    assert np.allclose(pt_b.grad.detach().cpu().numpy(), mir_b.grad), "Not equal b grad"
    assert np.allclose(pt_c.grad.detach().cpu().numpy(), mir_c.grad), "Not equal c grad"


def test_operation2():
    print("test_operation2")
    i, j, n, m = 3, 2, 3, 2

    pt_a = torch.rand((i, j, n, m), requires_grad=True)
    pt_b = torch.rand((1, n, m), requires_grad=True)
    pt_c = torch.rand((1, 1, n, m), requires_grad=True)
    pt_d = pt_a + pt_b ** 3
    pt_e = pt_d.exp()
    pt_f = pt_c.log() * pt_e.sin()
    pt_g = pt_f.sigmoid()
    pt_g.backward(torch.ones_like(pt_g))

    mir_a = Tensor(pt_a.detach().cpu().numpy())
    mir_b = Tensor(pt_b.detach().cpu().numpy())
    mir_c = Tensor(pt_c.detach().cpu().numpy())
    mir_d = mir_a + mir_b ** 3
    mir_e = mir_d.exp()
    mir_f = mir_c.log() * mir_e.sin()
    mir_g = mir_f.sigmoid()
    mir_g.backward()

    assert np.allclose(pt_a.grad.detach().cpu().numpy(), mir_a.grad), "Not equal a grad"
    assert np.allclose(pt_b.grad.detach().cpu().numpy(), mir_b.grad), "Not equal b grad"
    assert np.allclose(pt_c.grad.detach().cpu().numpy(), mir_c.grad), "Not equal c grad"


test_add()
test_mul()
test_matmul()
test_pow()
test_exp()
test_log()
test_sin()
test_cos()
test_sigmoid()
test_relu()
test_operation1()
test_operation2()