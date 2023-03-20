import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


class Plotter:
    def __init__(self, d4object):
        self.d4object = d4object
        self.fig, ((self.ax0, self.ax1), (self.ax2, self.ax3)) = plt.subplots(2,2,subplot_kw={"projection": "3d"},)

    def simulate(self):
        for iter in self.d4object.plot_single_section():

            self.ax0.set_title("x0")
            self.ax1.set_title("x1")
            self.ax2.set_title("x2")
            self.ax3.set_title("x3")

            self.ax0.set_xlim([-1, 1])
            self.ax0.set_ylim([-1, 1])
            self.ax0.set_zlim([-1, 1])
            
            self.ax1.set_xlim([-1, 1])
            self.ax1.set_ylim([-1, 1])
            self.ax1.set_zlim([-1, 1])

            self.ax2.set_xlim([-1, 1])
            self.ax2.set_ylim([-1, 1])
            self.ax2.set_zlim([-1, 1])

            self.ax3.set_xlim([-1, 1])
            self.ax3.set_ylim([-1, 1])
            self.ax3.set_zlim([-1, 1])

            self.ax0.scatter3D(iter[0][0], iter[0][1], iter[0][2], c=iter[0][2], cmap='Greens')
            self.ax1.scatter3D(iter[1][0], iter[1][1], iter[1][2], c=iter[1][2], cmap='Blues')
            self.ax2.scatter3D(iter[2][0], iter[2][1], iter[2][2], c=iter[2][2], cmap='bone')
            self.ax3.scatter3D(iter[3][0], iter[3][1], iter[3][2], c=iter[3][2], cmap='Purples')

            plt.pause(0.001)

            self.ax0.clear()
            self.ax1.clear()
            self.ax2.clear()
            self.ax3.clear()


class NDObject(ABC):
    def __init__(self, points:list):
        if not self.validate(points):
            raise ValueError('Points dimension different from declared one!')
        self.points = points

    @property
    @abstractmethod
    def DIM(self) -> int:
        raise NotImplementedError('Dimension has not been provided')

    def validate(self, points):
        return all(True if len(point) == self.DIM else False for point in points)

class D4Object(NDObject):
    SPHERE_MIN=-1
    SPHERE_MAX=1
    DIM=4
    def __init__(self, points):
        super().__init__(points)

    def plot_sections(self):
        pass

    def plot_single_section(self, threshold = 0.05):
        counter = self.SPHERE_MIN
        while counter <= self.SPHERE_MAX:
            result = {0:[], 1:[], 2:[], 3:[]}
            for axis_number in range(self.DIM):
                filtered_ = list(filter(lambda x: x[axis_number] >= counter and x[axis_number] <= counter + threshold, self.points))
                x1, x2, x3 = [], [], []
                if axis_number == 0:
                    ax1, ax2, ax3 = 1, 2, 3
                elif axis_number == 1:
                    ax1, ax2, ax3 = 0, 2, 3
                elif axis_number == 2:
                    ax1, ax2, ax3 = 0, 1, 3
                elif axis_number == 3:
                    ax1, ax2, ax3 = 0, 1, 2
                for point in filtered_:
                    x1.append(point[ax1])
                    x2.append(point[ax2])
                    x3.append(point[ax3])
                result[axis_number] = [x1,x2,x3]
            counter += threshold
            yield result

def sample_spherical(npoints, ndim=3):
    vec = np.random.randn(npoints, ndim)
    vec /= np.linalg.norm(vec, axis=0)
    return vec.T

if __name__ == '__main__':
    points= [point for point in sample_spherical(4,10000)]
    d4obj = D4Object(points)
    ploterInstance = Plotter(d4obj)
    ploterInstance.simulate()
