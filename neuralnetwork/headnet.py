import neuralnetwork.network as nn

class HeadNet(nn.network):
    def setup(self): (
        self.feed('input')
        .conv(3, 3, 1, 1, 8)
        .activate()
    )