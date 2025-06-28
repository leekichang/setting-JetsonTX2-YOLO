import torch
import torch.nn as nn

class EarlyExitLayer(nn.Module):
    def __init__(self, layers, num_classes, exit_layers=None, pool=True, dropout=True):
        super(EarlyExitLayer, self).__init__()
        self.layers = nn.Sequential(*layers) # feature extractor
        self.exit_layers = exit_layers
        if self.exit_layers is None:
            self.exit_layers = nn.Sequential()
        self.flatten_layers = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Flatten(),
        ) if pool else nn.Flatten()
        
        self.dropout = dropout
        self.is_initialized = False
        self.num_classes = num_classes
        
    def forward(self, x):
        if self.is_initialized:
            feature = self.layers(x)
            flatten_feature = self.flatten_layers(feature)
            preds = self.exit_layers(flatten_feature)

            return feature, flatten_feature, preds

        flatten_size = self.flatten_layers(self.layers(x)).shape[1]

        if self.dropout:
            self.exit_layers.add_module('dropout', nn.Dropout(0.5))
        self.exit_layers.add_module('fc', nn.Linear(flatten_size, self.num_classes))
        self.is_initialized = True

        return self.forward(x)
    


cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'F' : [64, 64, 'M', 128, 128,  'M', 256, 256, 'M', 512, 512, 'M'],
}


cfg_exitwise = {
    0: [64, 64, 'M'],
    1: [128, 128, 'M'],
    2: [256, 256, 'M'],
    3: [512, 512, 'M']
}

def make_layers(cfg, input_channel=1, batch_norm=False):
    layers = []

    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]
        if batch_norm:
            layers += [nn.BatchNorm2d(l)]
        layers += [nn.ReLU(inplace=True)]
        input_channel = l

    return nn.Sequential(*layers), input_channel

def make_ee_layers(cfg, num_classes, input_channel=3, batch_norm=False):
    layers = []

    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]
        if batch_norm:
            layers += [nn.BatchNorm2d(l)]
        layers += [nn.ReLU(inplace=True)]
        input_channel = l

    return EarlyExitLayer(layers, num_classes, pool=False, dropout=True), input_channel


class MultiExitVGG(nn.Module):
    def __init__(self, num_classes=10, exit_level=3):
        super().__init__()

        self.exit_level = exit_level
        self.num_classes = num_classes
        self.common, next_input_channel = make_layers(cfg_exitwise[0], batch_norm=True)
        self.exit1, next_input_channel = make_ee_layers(cfg_exitwise[1], num_classes, next_input_channel, batch_norm=True)
        self.exit2, next_input_channel = make_ee_layers(cfg_exitwise[2], num_classes, next_input_channel, batch_norm=True)
        self.exit3, next_input_channel = make_ee_layers(cfg_exitwise[3], num_classes, next_input_channel, batch_norm=True)
        self.layers = nn.ModuleList([self.exit1, self.exit2, self.exit3])

        self.initialized = False
        self._init_ee_layers()

    def _init_ee_layers(self):
        x = torch.zeros((32, 1, 3,151))
        self.eval()
        self.forward(x) # All ee layers are initialized automatically here.
        self.initialized = True

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.squeeze(0)
        x = x[:, :, :, :-1].reshape(x.size(0), 1, 18, 25)
        x = self.common(x)
        features = []
        preds = []

        for i, exit in enumerate(self.layers):
            x, flatten_feature, pred = exit(x)
            features.append(x)
            preds.append(pred)

            if self.initialized and self.exit_level == i + 1:
                return preds, features


class VGG(nn.Module):
    def __init__(self, cfg, num_classes=10):
        super().__init__()

        self.num_classes = num_classes
        self.features = make_layers(cfg, batch_norm=True)[0]

        self.classifier = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = x[:, :, :, :-1].reshape(x.size(0), 1, 18, 25)
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)

        return output

def vgg_bn(num_classes=8):
    return VGG(cfg['F'], num_classes=num_classes)

def multiexit_vgg_bn(num_classes=8, exit_level=3):
    return MultiExitVGG(num_classes=num_classes, exit_level=exit_level)

if __name__ == '__main__':
	import time

	x= torch.randn(100,1,3,151).to('cuda')
	model = vgg_bn().to('cuda')
	elapsed_times = []
	for i in range(100):
		with torch.no_grad():
			s = time.time()
			o = model(x)
			e = time.time()
		elapsed_times.append(e-s)
	import numpy as np
	print(f'{np.mean(elapsed_times):.4f}')
	import pickle
	with open('./time_test.pkl', 'wb') as f:
		pickle.dump(elapsed_times,f)
