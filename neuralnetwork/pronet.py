import neuralnetwork.network as nn

class pnet(nn.network):
    def setup(self):
        (
            self.feed('data')
            .conv(3, 3, 1, 1, 10, name='conv1', padding='VALID')
            .activate(name='relu1', activation='ReLU')
            .pool(2, 2, 2, 2, name='pool1', ptype_nn='MAX', padding='VALID')
            .conv(3, 3, 1, 1, 16, name='conv2', padding='VALID')
            .activate(name='relu2', activation='ReLU')
            .conv(3, 3, 1, 1, 32, name='conv3', padding='VALID')
            .activate(name='relu3', activation='ReLU')
            .conv(1, 1, 1, 1, 2, name='prob', padding='VALID')
            .softmax(-1, name='prob_sm')
        )
        (   self.feed('relu3')
            .conv(1, 1, 1, 1, 4, name='coor', padding='VALID')
        )

class rnet(nn.network):
    def setup(self):
        (
            self.feed('data')
            .conv(3, 3, 1, 1, 28, name='conv1', padding='VALID')
            .activate(name='relu1', activation='ReLU')
            .pool(3, 3, 2, 2, name='pool1', ptype_nn='MAX', padding='VALID')
            .conv(3, 3, 1, 1, 48, name='conv2', padding='VALID')
            .activate(name='relu2', activation='ReLU')
            .pool(3, 3, 2, 2, name='pool2', ptype_nn='MAX', padding='VALID')
            .conv(2, 2, 1, 1, 64, name='conv3', padding='VALID')
            .activate(name='relu3', activation='ReLU')
            .fc(128, name='fc1')
            .activate(name='relu4', activation='ReLU')
            .fc(2, name='prob')
            .softmax(1, name='prob_sm')
        )
        (
            self.feed('relu4')
            .fc(4, name='coor')
        )

class onet(nn.network):
    def setup(self):
        (
            self.feed('data')
            .conv(3, 3, 1, 1, 32, name='conv1', padding='VALID')
            .activate(name='prelu1', activation='PReLU')
            .pool(3, 3, 2, 2, name='pool1', ptype_nn='MAX', padding='VALID')
            .conv(3, 3, 1, 1, 64, name='conv2', padding='VALID')
            .activate(name='prelu2', activation='PReLU')
            .pool(3, 3, 2, 2, name='pool2', ptype_nn='MAX', padding='VALID')
            .conv(3, 3, 1, 1, 64, name='conv3', padding='VALID')
            .activate(name='prelu3', activation='PReLU')
            .pool(2, 2, 2, 2, name='pool3', ptype_nn='MAX', padding='VALID')
            .conv(2, 2, 1, 1, 128, name='conv4', padding='VALID')
            .activate(name='prelu4', activation='PReLU')
            .fc(256, name='fc1')
            .activate(name='prelu5', activation='PReLU')
            .fc(2, name='prob')
            .softmax(1, name='prob_sm')
        )
        (
            self.feed('prelu5')
            .fc(4, name='coor')
        )