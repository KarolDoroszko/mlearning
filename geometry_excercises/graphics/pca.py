from PIL import Image
from numpy import asarray
import numpy as np

class Picture:
    def __init__(self, path, output_path, output_path2, divider = 8, k = 7):
        self.path = path
        self.output_path = output_path
        self.output_path2 = output_path2
        self.divider = divider
        self.k = k
        self.image = None
        self.compressed_image = None
        self.get_picture()

    def adjust_shape(self):
        self.dim, self.number = int(self.dim/self.divider), int(self.number/self.divider)
        self.image = self.image[:self.dim * self.divider, :self.number * self.divider]
        self.whole_im_dim, self.whole_im_number = self.image.shape
        self.compressed_image = np.zeros((self.k * self.dim, self.number * self.divider))
        self.decompressed = np.zeros(self.image.shape)
        z= 0


    def iterate(self):
        for dim_part in range(self.dim):
            for quant_part in range(self.number):
                data_chunk = self.image[dim_part:dim_part + self.divider, quant_part:quant_part + self.divider]
                data_chunk, means  = Picture.center_data_chunk(data_chunk)
                data_chunk = np.dot(data_chunk, data_chunk.transpose())
                transformated_data_chunk, recovered_chunk = self.reduce_dim(data_chunk, means)
                self.compressed_image[dim_part*self.k:(dim_part +1)*self.k ,
                quant_part*self.divider:(quant_part+1)*self.divider] = transformated_data_chunk
                self.decompressed[dim_part*self.divider:(dim_part +1)*self.divider ,
                quant_part*self.divider:(quant_part+1)*self.divider] = recovered_chunk
                p = 0
            k=0
    def get_picture(self):
        image = Image.open(self.path)
        self.image = np.array(image.convert('L'))
        self.dim, self.number = self.image.shape

    @staticmethod
    def center_data_chunk(data_chunk):
        means = np.array([data_chunk.mean(axis = 0) for i in range(data_chunk.shape[1])])
        data_chunk = data_chunk - means
        return data_chunk, means

    def reduce_dim(self, data_chunk, means):
        eigen_vals, eigen_vectors = Picture.solve_eigen_problem(data_chunk)
        eigen_vals, eigen_vectors = eigen_vals, eigen_vectors[:,:self.k]
        new_chunk = eigen_vectors.real.transpose().dot(data_chunk)
        recovered_chunk = eigen_vectors.real.dot(new_chunk) + means
        return new_chunk, recovered_chunk

    @staticmethod
    def solve_eigen_problem(data_chunk):
        eigen_vals, eigen_vectors = np.linalg.eig(data_chunk)
        # eigen_vectors = eigen_vectors[0]
        idx = np.flip(eigen_vals.argsort()[::-1])
        eigen_vals, eigen_vectors = eigen_vals[idx], eigen_vectors[:, idx]
        return eigen_vals, eigen_vectors

    def save_compressed(self):
        # self.rescale_compressed()
        self.compressed_image = Image.fromarray(self.compressed_image.astype('uint8'))
        self.compressed_image.save(self.output_path)
        self.decompressed = Image.fromarray(self.decompressed.astype('uint8'))
        self.decompressed.save(self.output_path2)

    def rescale_compressed(self):
        x_min, x_max = self.compressed_image.min(), self.compressed_image.max()
        vfunc = np.vectorize(Picture.rescale)
        self.compressed_image = vfunc(x_min,x_max, self.compressed_image)
        z = 0

    @staticmethod
    def rescale(x_min, x_max, x):
        # x_min, x_max = self.compressed_image.min, self.compressed_image.max
        a = 255 / (x_max - x_min)
        b = -a * x_min
        return a * x + b


    def run(self):
        self.adjust_shape()
        self.iterate()
        z = 0
        self.save_compressed()


if __name__ == '__main__':
    # test_path = r"C:\Users\doros\Desktop\ZDJĘCIA\Zamowienie1_G.jpg"
    # test_path_out = r"C:\Users\doros\Desktop\ZDJĘCIA\Zamowienie1_G_compressed.jpg"
    test_path = r"C:\Users\doros\Desktop\ZDJĘCIA\Lena.png"
    test_path_out = r"C:\Users\doros\Desktop\ZDJĘCIA\Lena_compressed.png"
    test_path_out2 = r"C:\Users\doros\Desktop\ZDJĘCIA\Lena_decompressed.png"
    test_compr = Picture(test_path, test_path_out,test_path_out2)
    test_compr.run()

