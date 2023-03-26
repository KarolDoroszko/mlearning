import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
from pprint import pprint


class Arrow3D(FancyArrowPatch):

    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)

    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)


def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    '''Add an 3d arrow to an `Axes3D` instance.'''

    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)

setattr(Axes3D, 'arrow3D', _arrow3D)

class GradientDesc:
    def __init__(self,f,dim:int=3, max_iter:int = 50, dx:float =np.array([0.01, 0.01]), alpha:float = 0.1):
        self.f = f
        self.dim = dim
        self.max_iter = max_iter
        self.dx = dx
        self.alpha = alpha
        self.draw_x0()
        self.points = [self.xn]
        print(self.xn)

    def draw_x0(self):
        self.xn = np.random.uniform(low=9, high=10, size=(self.dim,))

    def calculate_grad(self):
        for k in range(self.max_iter):
            self.xn = self.xn - self.alpha * self.find_derivative()
            self.points.append(self.xn)
            print(self.xn)

    def find_derivative(self):
        result = []
        for d in range(self.dim):
            xn1 = np.array(self.xn)
            xn1[d] += self.dx[d]
            result.append((self.f(*xn1) - self.f(*self.xn)) / self.dx[d])
        return np.array(result)

    def plot_points(self, range_ = [[-11,11], [-11, 11]]):
        fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"}, )
        x_min, x_max = range_[0]
        y_min, y_max = range_[1]
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_zlim([-1, x_max*y_max+100])
        X = np.arange(x_min, x_max, self.dx[0])
        Y = np.arange(y_min, y_max, self.dx[1])
        X, Y = np.meshgrid(X, Y)
        Z = self.f(X,Y)
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,linewidth=0, antialiased=False)
        for index in range(len(self.points) -1):
            start_x, start_y = self.points[index]
            stop_x, stop_y = self.points[index + 1] - self.points[index]
            start_z, stop_z = self.f(*self.points[index]), (self.f(*self.points[index+1]) - self.f(*self.points[index]) )
            ax.arrow3D(start_x, start_y, start_z,stop_x, stop_y, stop_z, mutation_scale=10, arrowstyle="-|>", linestyle='dashed')

        plt.show()

if __name__ == '__main__':
    def f_x(x,y):
        return x**2 + 2 * y*3

    grad_desc_test = GradientDesc(f_x,dim=2)
    grad_desc_test.calculate_grad()
    grad_desc_test.plot_points()

