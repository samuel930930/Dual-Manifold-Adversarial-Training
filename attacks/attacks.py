

def inverse_transform(x):
    # [-1, 1] -> [0, 255]
    x = x * 0.5 + 0.5
    return x * 255.


def transform(x):
    # [0, 255] -> [-1, 1]
    x = x / 255.
    return x * 2 - 1


class PixelModel(object):
    def __init__(self, model):
        self.model = model

    def __call__(self, x):
        x = transform(x)
        x = self.model(x)
        return x
