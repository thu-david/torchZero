import numpy as np


class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]

        while funcs:
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)
            if x.creator is not None:
                funcs.append(x.creator)


class Function:
    def __call__(self, inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(xs)
        self.inputs = inputs
        outputs = [Variable(self.as_array(y)) for y in ys]

        for output in outputs:
            output.set_creator(self)
        self.outputs = outputs
        return outputs

    def forward(self, xs):
        raise NotImplementedError

    def backward(self, gys):
        raise NotImplementedError

    def as_array(self, x):
        if np.isscalar(x):
            return np.array(x)
        return x


class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx


class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx


def square(x):
    return Square()(x)


def exp(x):
    return Exp()(x)


class Add(Function):
    def forward(self, xs):
        x0, x1 = xs
        y = x0 + x1
        return (y, )


# def numerical_diff(f, x, eps=1e-4):
#     x0 = Variable(x.data - eps)
#     x1 = Variable(x.data + eps)
#     y0 = f(x0)
#     y1 = f(x1)
#     return (y1.data - y0.data) / (2 * eps)
#
#
# class SquareTest(unittest.TestCase):
#     def test_forward(self):
#         x = Variable(np.array(2.0))
#         y = square(x)
#         expected = np.array(4.0)
#         self.assertEqual(expected, y.data)
#
#     def test_gradient_check(self):
#         x = Variable(np.random.rand(1))
#         y = square(x)
#         y.backward()
#         num_grad = numerical_diff(square, x)
#         flag = np.allclose(num_grad, x.grad)
#         self.assertTrue(flag)


if __name__ == '__main__':
    xs = [Variable(np.array(3.0)), Variable(np.array(2.0))]
    f = Add()
    ys = f(xs)
    y = ys[0]
    print(y.data)

