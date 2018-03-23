import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import OrderedDict

class NeuralNet(nn.Module):
    #def __init__(self, numberOfConvolutionKernelsList, maxPoolingKernelList, numberOfInputChannels,
    def __init__(self, numberOfConvolutions_KernelSize_Pooling_List, numberOfInputChannels,
                 classesNbr, imageSize, dropoutRatio=0.5):
        super(NeuralNet, self).__init__()

        if len(numberOfConvolutions_KernelSize_Pooling_List) < 1:
            raise RuntimeError("ConvStackClassifier.NeuralNet.__init__(): The number of convolution layers is 0")
        self.lastLayerImageSize = imageSize
        layersDict = OrderedDict()

        layersDict['conv0'] = nn.Sequential(
            nn.Conv2d(numberOfInputChannels, numberOfConvolutions_KernelSize_Pooling_List[0][0],
                      numberOfConvolutions_KernelSize_Pooling_List[0][1],
                      padding=(int) (numberOfConvolutions_KernelSize_Pooling_List[0][1]/2) ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(numberOfConvolutions_KernelSize_Pooling_List[0][2]))
        self.lastLayerImageSize = (int) (self.lastLayerImageSize/ numberOfConvolutions_KernelSize_Pooling_List[0][2])
        for layerNdx in range(1, len(numberOfConvolutions_KernelSize_Pooling_List)):
            layersDict['conv{}'.format(layerNdx)] = nn.Sequential(
                nn.Conv2d(numberOfConvolutions_KernelSize_Pooling_List[layerNdx - 1][0], numberOfConvolutions_KernelSize_Pooling_List[layerNdx][0],
                          numberOfConvolutions_KernelSize_Pooling_List[layerNdx][1],
                          padding=(int)(numberOfConvolutions_KernelSize_Pooling_List[layerNdx][1] / 2)),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(numberOfConvolutions_KernelSize_Pooling_List[layerNdx][2]))
            self.lastLayerImageSize = (int)(self.lastLayerImageSize / numberOfConvolutions_KernelSize_Pooling_List[layerNdx][2])

        self.convLayers = nn.Sequential(layersDict)
        self.lastLayerNumberOfChannels = numberOfConvolutions_KernelSize_Pooling_List[-1][0]
        self.numberOfConvolutionLayers = len(layersDict)
        self.linear1 = nn.Linear(self.lastLayerImageSize * self.lastLayerImageSize *
                                 self.lastLayerNumberOfChannels, classesNbr)
        self.dropout = nn.Dropout2d(p=dropoutRatio)


    def forward(self, inputs):
        activation = self.convLayers[0](inputs)
        for layerNdx in range(1, self.numberOfConvolutionLayers):
            activation = self.convLayers[layerNdx](activation)

        vector = activation.view(-1, self.lastLayerImageSize * self.lastLayerImageSize * self.lastLayerNumberOfChannels)
        drop = self.dropout(vector)
        outputLin = self.linear1(drop)
        return F.log_softmax(outputLin)
