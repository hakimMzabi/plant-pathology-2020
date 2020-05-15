import matplotlib.pyplot as plt
from src.helper import Helper


class PlantPathology:
    """
    Ease CIFAR-10 creation with indicated shape
    """

    def __init__(self, dim=1):
        self.helper = Helper()
        if 1 <= dim <= 3:
            (self.train_generator, self.valid_generator,
             self.test_generator) = self.helper.get_plant_pathology_prepared()
        else:
            raise Exception("Plant Pathology dataset couldn't be initialized with dims != 3 or != 1")

    def plot_image(self, nb_of_images):
        sample_training, _ = next(self.train_generator)
        images = sample_training[:nb_of_images]
        fig, axes = plt.subplots(1, 5, figsize=(20, 20))
        axes = axes.flatten()
        for img, ax in zip(images, axes):
            print(img.shape)
            ax.imshow(img)
            ax.axis('off')
        plt.tight_layout()
        plt.show()

    def step_train(self):
        return self.train_generator.n // self.train_generator.batch_size

    def step_valid(self):
        return self.valid_generator.n // self.valid_generator.batch_size

    def step_test(self):
        return self.test_generator.n // self.test_generator.batch_size
