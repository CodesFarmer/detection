import neuralnetwork.network as nn

width_multiplier = 0.25
min_depth = 16
class MobileNetV1(nn.network):
    def setup(self): (
        self.feed('input')
        .conv(3, 3, 2, 2, 32, 'conv1', padding='SAME')
        .mobile_unit(64, 'mbn1', strdies=[1, 1, 1, 1], padding='SAME',
                     width_multiplier=width_multiplier, min_depth=min_depth)
        .mobile_unit(128, 'mbn2', strdies=[1, 2, 2, 1], padding='SAME',
                     width_multiplier=width_multiplier, min_depth=min_depth)
        .mobile_unit(128, 'mbn3', strdies=[1, 1, 1, 1], padding='SAME',
                     width_multiplier=width_multiplier, min_depth=min_depth)
        .mobile_unit(256, 'mbn4', strdies=[1, 2, 2, 1], padding='SAME',
                     width_multiplier=width_multiplier, min_depth=min_depth)
        .mobile_unit(256, 'mbn5', strdies=[1, 1, 1, 1], padding='SAME',
                     width_multiplier=width_multiplier, min_depth=min_depth)
        .mobile_unit(512, 'mbn6', strdies=[1, 2, 2, 1], padding='SAME',
                     width_multiplier=width_multiplier, min_depth=min_depth)
        .mobile_unit(512, 'mbn7', strdies=[1, 1, 1, 1], padding='SAME',
                     width_multiplier=width_multiplier, min_depth=min_depth)
        .mobile_unit(512, 'mbn8', strdies=[1, 1, 1, 1], padding='SAME',
                     width_multiplier=width_multiplier, min_depth=min_depth)
        .mobile_unit(512, 'mbn9', strdies=[1, 1, 1, 1], padding='SAME',
                     width_multiplier=width_multiplier, min_depth=min_depth)
        .mobile_unit(512, 'mbn10', strdies=[1, 1, 1, 1], padding='SAME',
                     width_multiplier=width_multiplier, min_depth=min_depth)
        .mobile_unit(512, 'mbn11', strdies=[1, 1, 1, 1], padding='SAME',
                     width_multiplier=width_multiplier, min_depth=min_depth)
        .mobile_unit(1024, 'mbn12', strdies=[1, 2, 2, 1], padding='SAME',
                     width_multiplier=width_multiplier, min_depth=min_depth)
        .mobile_unit(1024, 'mbn13', strdies=[1, 1, 1, 1], padding='SAME',
                     width_multiplier=width_multiplier, min_depth=min_depth)
    )