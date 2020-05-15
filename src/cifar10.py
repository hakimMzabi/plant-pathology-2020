from src.helper import Helper

class Cifar10:
    """
    Ease CIFAR-10 creation with indicated shape
    """
    def __init__(self, dim=1):
        self.helper = Helper()
        if dim >= 1 and dim <= 3:
            (self.x_train, self.y_train), (self.x_test, self.y_test) = self.helper.get_cifar10_prepared(dim=dim)
        else:
            raise Exception("Cifar 10 couldn't be initialized with dims != 3 or != 1")
