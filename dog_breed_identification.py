import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import argparse
import pandas
import PIL
import os
import numpy
import ConvStackClassifier
import cnn_finetune

print("dog_breed_identification")

parser = argparse.ArgumentParser()
parser.add_argument('baseDirectory', help='The directory containing the train and test folders')
parser.add_argument('labelsPlusTrainOrValid', help='The csv file with 3 columns: id, breed, usage (train or valid)')
parser.add_argument('--numberOfConvolutionLayers', help='The number of convolution layers', type=int, default=3)
parser.add_argument('--numberOfKernelsPerLayer', help='The number of convolution kernels of each layer', type=int, default=32)
parser.add_argument('--imageSize', help='The (square) size the images will be resized to', type=int, default=224)
parser.add_argument('--maximumNumberOfTrainingImages', help='The maximum number of training images to load', type=int, default=0)
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--learningRate', help='The learning rate', type=float, default=0.001)
parser.add_argument('--momentum', help='The learning momentum', type=float, default=0.9)
parser.add_argument('--dropoutRatio', help='The dropout ratio', type=float, default=0.5)
parser.add_argument('--saveDirectory', help='The directory where the files will be saved', default='/tmp')
parser.add_argument('--structure', help='The neural network structure. Ex.: ConvStack_1_2_32_7_2_32_7_2_10_28_0.5')

args = parser.parse_args()
args.cuda = not args.disable_cuda and torch.cuda.is_available()

# -------------------- Utilities -------------------------------
def MostProbableClass(outputVector):
    mostProbableClass = -1;
    highestOutput = -float('inf')
    for classNdx in range (outputVector.shape[0]):
        if outputVector[classNdx] > highestOutput:
            highestOutput = outputVector[classNdx]
            mostProbableClass = classNdx
    return mostProbableClass
# ----------------------------------------------------------------


# Load the labels file
idLabelUsageDataFrame = pandas.read_csv(args.labelsPlusTrainOrValid)
# Make lists of filenames for train and validation
trainFileIDs = []
validationFileIDs = []
for index, row in idLabelUsageDataFrame.iterrows():
    #print ("index: {}; row = {}".format(index, row))
    if row['usage'] == 'train':
        trainFileIDs.append(row['id'])
    elif row['usage'] == 'valid':
        validationFileIDs.append(row['id'])
    else:
        raise KeyError("Unknown usage '{}'".format(row['usage']))

if args.maximumNumberOfTrainingImages <= 0 or args.maximumNumberOfTrainingImages > len(trainFileIDs):
    args.maximumNumberOfTrainingImages = len(trainFileIDs)

#print("len(trainFileIDs = {}".format(len(trainFileIDs)))
#print("len(validationFileIDs = {}".format(len(validationFileIDs)))

# Count the number of breeds
#breedFrqDataFrame = idLabelUsageDataFrame.breed.value_counts()
# scottish_deerhound                126
# maltese_dog                       117
# ...
numberOfBreeds = idLabelUsageDataFrame.breed.nunique()
print("numberOfBreeds = {}".format(numberOfBreeds))
#print ("breedFrqDataFrame = {}".format(breedFrqDataFrame))
breedsList = idLabelUsageDataFrame.breed.unique()
#print("breedsList = {}".format(breedsList))

def BreedIndex(breed, breedsList):
    foundIndex = -1
    for index in range(len(breedsList)):
        if breedsList[index] == breed:
            foundIndex = index
    if foundIndex == -1:
        raise RuntimeError("BreedIndex(): Could not find breed '{}'".format(breed))
    return foundIndex


# Preprocessing
normalize = torchvision.transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
preprocess = torchvision.transforms.Compose([
    torchvision.transforms.Scale(256),
    torchvision.transforms.Resize((args.imageSize, args.imageSize)),
    torchvision.transforms.ToTensor(),
    normalize
])

trainTensor = torch.Tensor(args.maximumNumberOfTrainingImages, 3, args.imageSize, args.imageSize)
#trainLabelTensor = torch.LongTensor(args.maximumNumberOfTrainingImages, numberOfBreeds).zero_()
trainLabelTensor = torch.LongTensor(args.maximumNumberOfTrainingImages).zero_()

for trainExampleNdx in range(args.maximumNumberOfTrainingImages):
    imageFilepath = os.path.join(args.baseDirectory, "train", trainFileIDs[trainExampleNdx] + ".jpg")

    trainImage = PIL.Image.open(imageFilepath)
    if trainExampleNdx == 0:
        print("imageFilepath = {}".format(imageFilepath))
        #trainImage.show()
    #print("trainImage.size = {}".format(trainImage.size))
    img_tensor = preprocess(trainImage)
    #print("img_tensor.shape = {}".format(img_tensor.shape))
    img_tensor.unsqueeze_(0) # img_tensor.shape: torch.Size([3, 224, 224]) -> torch.Size([1, 3, 224, 224])
    # Put it in the tensor
    trainTensor[trainExampleNdx] = img_tensor

    # Find the breed (class) index
    dataLines = idLabelUsageDataFrame.loc[idLabelUsageDataFrame['id'] == trainFileIDs[trainExampleNdx] ]
    if len(dataLines) != 1:
        raise RuntimeError("The number of entries whose 'id' is '{}' is not 1 ({})".format(trainFileIDs[trainExampleNdx], len(dataLines)))
    #print(dataLines) # id breed usage
    breedIndex = -1
    for index, row in dataLines.iterrows():
        breedIndex = BreedIndex(row['breed'], breedsList)
        #print("index = {}; row['breed'] = {}; breedIndex = {}".format(index, row['breed'], breedIndex))

    trainLabelTensor[trainExampleNdx] = breedIndex

    if trainExampleNdx % 20 == 0:
        print("trainExampleNdx {}/{}".format(trainExampleNdx, args.maximumNumberOfTrainingImages) )

numberOfValidationImages = min( int(args.maximumNumberOfTrainingImages * len(validationFileIDs)/len(trainFileIDs) ), len(validationFileIDs))
validationTensor = torch.Tensor(numberOfValidationImages, 3, args.imageSize, args.imageSize)
#validationLabelTensor = torch.LongTensor(numberOfValidationImages, numberOfBreeds).zero_()
validationLabelTensor = torch.LongTensor(numberOfValidationImages).zero_()

for validationExampleNdx in range(numberOfValidationImages):
    imageFilepath = os.path.join(args.baseDirectory, "train", validationFileIDs[validationExampleNdx] + ".jpg")

    validationImage = PIL.Image.open(imageFilepath)
    if validationExampleNdx == 0:
        print("imageFilepath = {}".format(imageFilepath))
        # trainImage.show()

    img_tensor = preprocess(validationImage)
    img_tensor.unsqueeze_(0)  # img_tensor.shape: torch.Size([3, 224, 224]) -> torch.Size([1, 3, 224, 224])
    # Put it in the tensor
    validationTensor[validationExampleNdx] = img_tensor

    # Find the breed (class) index
    dataLines = idLabelUsageDataFrame.loc[idLabelUsageDataFrame['id'] == validationFileIDs[validationExampleNdx]]
    if len(dataLines) != 1:
        raise RuntimeError(
            "The number of entries whose 'id' is '{}' is not 1 ({})".format(validationFileIDs[validationExampleNdx],
                                                                            len(dataLines)))
    # print(dataLines) # id breed usage
    breedIndex = -1
    for index, row in dataLines.iterrows():
        breedIndex = BreedIndex(row['breed'], breedsList)
        # print("index = {}; row['breed'] = {}; breedIndex = {}".format(index, row['breed'], breedIndex))

    validationLabelTensor[validationExampleNdx] = breedIndex

    if validationExampleNdx % 20 == 0:
        print("validationExampleNdx {}/{}".format(validationExampleNdx, numberOfValidationImages))


#print ("validationLabelTensor = {}".format(validationLabelTensor))

# Create a neural network
if args.structure is not None:
    neuralNet = ConvStackClassifier.NeuralNet(structure=args.structure)

else:
    neuralNet = cnn_finetune.make_model(
        'resnet50',
        pretrained=True,
        num_classes=numberOfBreeds,
        dropout_p=0.5,
    )
    """
    numberOfConvolutionKernelsList = []
    maxPoolingKernelList = []
    for layerNdx in range(args.numberOfConvolutionLayers):
        numberOfConvolutionKernelsList.append(args.numberOfKernelsPerLayer)
        maxPoolingKernelList.append(2)

    neuralNet = ConvStackClassifier.NeuralNet(numberOfConvolutionKernelsList, maxPoolingKernelList,
                                              numberOfBreeds, args.imageSize,
                                              args.dropoutRatio)
    """
if args.cuda:
    neuralNet.cuda() # Move to GPU


optimizer = torch.optim.SGD(neuralNet.parameters(), lr=args.learningRate, momentum=args.momentum)
lossFunction = nn.NLLLoss()

def MinibatchIndices(numberOfSamples, minibatchSize):
    shuffledList = numpy.arange(numberOfSamples)
    numpy.random.shuffle(shuffledList)
    minibatchesIndicesList = []
    numberOfWholeLists = int(numberOfSamples / minibatchSize)
    for wholeListNdx in range(numberOfWholeLists):
        minibatchIndices = shuffledList[ wholeListNdx * minibatchSize : (wholeListNdx + 1) * minibatchSize ]
        minibatchesIndicesList.append(minibatchIndices)
    # Add the last incomplete minibatch
    if numberOfWholeLists * minibatchSize < numberOfSamples:
        lastMinibatchIndices = shuffledList[numberOfWholeLists * minibatchSize:]
        minibatchesIndicesList.append(lastMinibatchIndices)
    return minibatchesIndicesList


if args.cuda:
    validationTensor = validationTensor.cuda()
if args.cuda:
    validationLabelTensor = validationLabelTensor.cuda()

minibatchSize = 64
minibatchIndicesListList = MinibatchIndices(args.maximumNumberOfTrainingImages, minibatchSize)

trainingDataFilepath = os.path.join(args.saveDirectory, 'trainingEpochs.csv')
trainingDataFile = open(trainingDataFilepath, "w")
trainingDataFile.write('epoch,averageTrainLoss,validationLoss\n')

#print ("minibatchIndicesListList = {}".format(minibatchIndicesListList))
for epoch in range(200):
    averageTrainLoss = 0
    for minibatchListNdx in range(len(minibatchIndicesListList)):
        minibatchIndicesList = minibatchIndicesListList[minibatchListNdx]
        thisMinibatchSize = len(minibatchIndicesList)

        minibatchInputImagesTensor = torch.autograd.Variable(
            torch.index_select(trainTensor, 0, torch.LongTensor(minibatchIndicesList)))
        minibatchTargetOutputTensor = torch.autograd.Variable(
            torch.index_select(trainLabelTensor, 0, torch.LongTensor(minibatchIndicesList)))
        if args.cuda:
            minibatchInputImagesTensor = minibatchInputImagesTensor.cuda()
            minibatchTargetOutputTensor = minibatchTargetOutputTensor.cuda()
            neuralNet.cuda()

        # Zero gradients
        optimizer.zero_grad()
        # Forward pass
        actualOutput = neuralNet(minibatchInputImagesTensor)
        # Loss
        """targetOutputShape = minibatchTargetOutputTensor.data.shape
        actualOutputShape = actualOutput.data.shape
        print("targetOutputShape = {}; actualOutputShape = {}".format(targetOutputShape, actualOutputShape))
        """
        loss = lossFunction(actualOutput, minibatchTargetOutputTensor)
        # if minibatchNdx == 0:
        #    print("Train loss = {}".format(loss.data[0]))

        # Backward pass
        loss.backward()
        # Parameters update
        optimizer.step()

        averageTrainLoss += loss.data[0]

    averageTrainLoss = averageTrainLoss / len(minibatchIndicesListList)

    # Validation loss
    validationOutput = neuralNet( torch.autograd.Variable(validationTensor) )
    validationLoss = lossFunction(validationOutput, torch.autograd.Variable(
        validationLabelTensor) )

    # Accuracy
    numberOfCorrectPredictions = 0
    for validationNdx in range(numberOfValidationImages):
        mostProbableClass = MostProbableClass(validationOutput[validationNdx].data)
        if mostProbableClass == validationLabelTensor[validationNdx]:
            numberOfCorrectPredictions += 1
    accuracy = numberOfCorrectPredictions / numberOfValidationImages

    print("Epoch {}: Average train loss = {}; validationLoss = {}; accuracy = {}".format(epoch, averageTrainLoss,
                                                                                         validationLoss.data[0],
                                                                                         accuracy))
    #neuralNet.Save(args.saveDirectory, str(validationLoss.data[0]))
    trainingDataFile.write("{},{},{}\n".format(epoch, averageTrainLoss, validationLoss.data[0]))

trainingDataFile.close()