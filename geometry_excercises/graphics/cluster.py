import numpy as np
from PIL import Image

class Picture:
    def __init__(self,picture_path:np.array, output_path: str,  new_dim_k: int = 3):
        self.picture, self.segmented = None, None
        self.read_image(picture_path)
        self.output_path = output_path
        self.new_dim_k = new_dim_k
        self.min = self.picture.min()
        self.max = self.picture.max()
        self.representatives = np.zeros(new_dim_k)
        self.class_cardinalities = np.zeros(new_dim_k)
        self.error = None
        self.iter_stop = False
        self.initialise()
        self.counter = 0

    def read_image(self, picture_path):
        self.picture = Image.open(picture_path)
        self.picture = np.array(self.picture.convert('L'))
        self.segmented = np.array(self.picture)

    def save_image(self,):
        self.segmented = Image.fromarray(self.segmented.astype('uint8'))
        self.segmented.save(self.output_path)

    def initialise(self):
        self.representatives = np.random.uniform(self.min, self.max, self.new_dim_k)
        # mu = (self.max + self.min) / 2
        # std = (self.max - self.min) /3
        # self.representatives = np.random.uniform(mu, std, self.new_dim_k)

    def split_again(self):
        self.representatives = np.array([self.picture[np.where(self.segmented == k_class)].mean()
                                         for k_class in self.representatives])

    def check_stop(self, new_class_cardinalities, segmented):
        if np.array_equal(self.segmented, segmented):
            self.iter_stop = True
            self.save_image()
        else:
            self.segmented = segmented
            self.class_cardinalities = new_class_cardinalities
            self.split_again()

    def cluster(self):
        while self.iter_stop == False:
            print(self.counter)
            self.counter += 1
            self.classify()

    @staticmethod
    def calculate_points_dist(p1, p2):
        return abs(p2-p1)

    def classify_point(self, p1):
        classification = np.array([Picture.calculate_points_dist(k_class, p1) for k_class in self.representatives])
        return self.representatives[np.where(classification == classification.min())]

    def classify(self):
        apply_class = np.vectorize(self.classify_point)
        segmented = apply_class(self.picture)
        new_class_cardinalities = np.array([np.count_nonzero(segmented == class_) for class_ in self.representatives])
        self.check_stop(new_class_cardinalities, segmented)


if __name__ == '__main__':
    test_path = r"C:\Users\doros\Desktop\ZDJĘCIA\Zamowienie1_G.jpg"
    output_path = r"C:\Users\doros\Desktop\ZDJĘCIA\Zamowienie1_G_postprocessed_segmented.jpg"
    image = Image.open(test_path)
    data = np.array(image.convert('L'))
    picture = Picture(test_path, output_path, 9)
    picture.cluster()


