import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class NeuralNet(nn.Module):
    def __init__(self, numberOfConvolutionKernelsList, maxPoolingKernelList, classesNbr,
                 imageSize, dropoutRatio=0.5):
        super(NeuralNet, self).__init__()
        if len(numberOfConvolutionKernelsList) < 1:
            raise RuntimeError("ConvStackClassifier.NeuralNet.__init__(): numberOfConvolutionKernelsList is empty")
        if len(numberOfConvolutionKernelsList) != len(maxPoolingKernelList):
            raise RuntimeError("ConvStackClassifier.NeuralNet.__init__(): len(numberOfConvolutionKernelsList) ({}) != len(maxPoolingKernelList) ({})".format(len(numberOfConvolutionKernelsList), len(maxPoolingKernelList)) )

        #self.convLayers = []
        poolingReductionFactor = maxPoolingKernelList[0]
        self.add_module('conv0', nn.Sequential(
            nn.Conv2d(3, numberOfConvolutionKernelsList[0], kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(maxPoolingKernelList[0])))
        for layerNdx in range(1, len(numberOfConvolutionKernelsList)):
            self.add_module('conv{}'.format(layerNdx), nn.Sequential(
                nn.Conv2d(numberOfConvolutionKernelsList[layerNdx - 1], numberOfConvolutionKernelsList[layerNdx], kernel_size=7, padding=3),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(maxPoolingKernelList[layerNdx])))
            poolingReductionFactor *= maxPoolingKernelList[layerNdx]
        self.lastLayerImageSize = int(imageSize / poolingReductionFactor)
        self.lastLayerNumberOfChannels = numberOfConvolutionKernelsList[-1]
        self.linear1 = nn.Linear(self.lastLayerImageSize * self.lastLayerImageSize *
                                 self.lastLayerNumberOfChannels, classesNbr)
        self.dropout = nn.Dropout2d(p=dropoutRatio)
        self.numberOfConvolutionLayers = len(numberOfConvolutionKernelsList)

    def forward(self, inputs):
        #moduleNames, modules = self.named_childer()
        activation = self.conv0(inputs)
        for layerNdx in range(1, self.numberOfConvolutionLayers):
            layerName = 'conv{}'.format(layerNdx)
            # Find the module by its name
            for name, module in self.named_children():
                if name == layerName:
                    activation = module(activation)
            #activation = self.convLayers[layerNdx](activation)

        vector = activation.view(-1, self.lastLayerImageSize * self.lastLayerImageSize * self.lastLayerNumberOfChannels)
        drop = self.dropout(vector)
        outputLin = self.linear1(drop)
        return F.log_softmax(outputLin, dim=0)
