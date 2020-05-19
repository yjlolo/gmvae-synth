class BaseLoss:
    def __init__(self, weight=1, effect_epoch=1):
        self.weight = weight
        self.effect_epoch = effect_epoch

    def __call__(self):
        raise NotImplementedError
