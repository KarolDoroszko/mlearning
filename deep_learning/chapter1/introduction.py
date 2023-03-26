import numpy as np

def cost_function(x,y):
    return abs(x-y) + abs(x-1) - abs(y)

def minx(x):
    return x/2

def miny(y):
    return (y+1)/2

def do_iteration(v_i):
    x_i, y_i = v_i
    y_i1 = minx(x_i)
    x_i1 = miny(y_i1)
    return (x_i1,y_i1)

if __name__ == '__main__':
    points = [(x**2, 0.5 * y**2) for x in np.arange(-10,10,0.1) for y in np.arange(-10,10,0.1)]
    z = points[150]
    print(z)

    threshold = 0.000001
    while threshold:
        new_z = do_iteration(z)
        length = abs((new_z[0]-z[0])**2 + (new_z[1]-z[1])**2)
        print(f'({z[0]},{z[1]}), ({new_z[0]}, {new_z[1]}),{length}')
        if length <threshold:
            threshold = False
        else:
            z = new_z
