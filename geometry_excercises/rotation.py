import numpy as np

class Gobject:
    def __init__(self, points:list):
        self.points = points

    def rotate(self, angles):
        rotation = Gobject.get_rotation_operator(angles)
        self.points = list(map(lambda vect: rotation.dot(vect), self.points))

    def __str__(self):
        result = '\n'.join(f'x = {point[0]}, y = {point[1]}, z = {point[2]}' for point in self.points)
        return result

    @staticmethod
    def get_rotation_operator(angles:list):
        alpha,beta,gamma = angles
        rx = np.array([[1,0,0],
                       [0,np.cos(alpha), -np.sin(alpha)],
                       [0,np.sin(alpha), np.cos(alpha)]])
        ry = np.array([[np.cos(beta), 0, np.sin(beta)],
                       [0,1,0],
                       [-np.sin(beta), 0, np.cos(beta)]])
        rz = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                       [np.sin(gamma), np.cos(gamma), 0],
                       [0,0,1]])

        r = np.matmul(np.matmul(rx,ry),rz)
        return r


if __name__ == '__main__':
    a = np.array([1,1,1])
    b = np.array([1,1,0])
    angles = [np.pi/2, 0, 0]
    test_object = Gobject([a,b])
    test_object.rotate(angles)
    print(test_object)