from PIL import Image
from numpy import asarray
import numpy as np

def extend_array(image:np.array, filter_size = 2):
    shape_y, shape_x = image.shape
    row_to_add = np.array([0 for item in range(shape_x)])
    column_to_add = np.array([0 for item in range(shape_y)])
    for counter in range(filter_size):
        image = np.insert(image, shape_x + counter, column_to_add, axis=1)
        row_to_add = np.append(row_to_add,[0])
        image = np.insert(image, 0, column_to_add, axis=1)
        row_to_add = np.append(row_to_add, [0])
        image = np.insert(image, shape_y + counter, row_to_add, axis=0)
        column_to_add = np.append(column_to_add, [0])
        image = np.insert(image, 0 + counter, row_to_add, axis=0)
        column_to_add = np.append(column_to_add, [0])
    return image

def filter(image:np.array, filter_size:int = 9, filter_vlues: np.array=1/9*np.array([[1,0,1],[0,1,0],[1,0,1]])):
    #TO DO: VALIDATION OF INPUT
    shape_y, shape_x = image.shape
    result = np.zeros(image.shape)
    filter_size = int(np.sqrt(filter_size))
    adjustment = int( (filter_size -1) / 2  ) if filter_size !=2 else 1
    image = extend_array(image, adjustment)
    for i in range(shape_x):
        for j in range(shape_y):
            result[j,i] = sum(sum(image[j:j+filter_size,i:i+filter_size]* filter_vlues))
    return result

if __name__ == '__main__':
    test_path = r"C:\Users\doros\Desktop\ZDJĘCIA\Zamowienie1_G.jpg"
    image = Image.open(test_path)
    print(image.format)
    print(image.size)
    print(image.mode)
    data = np.array(image.convert('L'))
    new_data = filter(data, 4, np.array([[1,0],[0,-1]]))
    new_data2 = filter(data, 4, np.array([[0,1],[-1,0]]))
    new_data = abs(new_data) + abs(new_data2)
    new_im = Image.fromarray(new_data.astype('uint8'))
    new_im.save(r"C:\Users\doros\Desktop\ZDJĘCIA\Zamowienie1_G_postprocessed_bond.jpg")
