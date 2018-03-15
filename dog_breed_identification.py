import torch
import torchvision
import argparse
import pandas
import PIL
import os

print("dog_breed_identification")

parser = argparse.ArgumentParser()
parser.add_argument('baseDirectory', help='The directory containing the train and test folders')
parser.add_argument('labelsPlusTrainOrValid', help='The csv file with 3 columns: id, breed, usage (train or valid)')
parser.add_argument('--imageSize', help='The size cropped from a 256 x 256 image', type=int, default=224)
parser.add_argument('--maximumNumberOfTrainingImages', help='The maximum number of training images to load', type=int, default=0)
args = parser.parse_args()

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

if args.maximumNumberOfTrainingImages <= 0:
    args.maximumNumberOfTrainingImages = len(trainFileIDs)

#print("len(trainFileIDs = {}".format(len(trainFileIDs)))
#print("len(validationFileIDs = {}".format(len(validationFileIDs)))

# Count the number of breeds
#breedFrqDataFrame = idLabelUsageDataFrame.breed.value_counts()
# scottish_deerhound                126
# maltese_dog                       117
# ...
numberOfBreeds = idLabelUsageDataFrame.breed.nunique()
#print("numberOfBreeds = {}".format(numberOfBreeds))
#print ("breedFrqDataFrame = {}".format(breedFrqDataFrame))
breedsList = idLabelUsageDataFrame.breed.unique()
print("breedsList = {}".format(breedsList))

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
    torchvision.transforms.CenterCrop(args.imageSize),
    torchvision.transforms.ToTensor(),
    normalize
])



trainTensor = torch.Tensor(args.maximumNumberOfTrainingImages, 3, args.imageSize, args.imageSize)
trainLabelTensor = torch.Tensor(args.maximumNumberOfTrainingImages, numberOfBreeds).zero_()

for trainExampleNdx in range(args.maximumNumberOfTrainingImages):
    imageFilepath = os.path.join(args.baseDirectory, "train", trainFileIDs[trainExampleNdx] + ".jpg")

    trainImage = PIL.Image.open(imageFilepath)
    if trainExampleNdx == 0:
        print("imageFilepath = {}".format(imageFilepath))
        #trainImage.show()

    img_tensor = preprocess(trainImage)
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

    trainLabelTensor[trainExampleNdx, breedIndex] = 1.0

    if trainExampleNdx % 20 == 0:
        print("trainExampleNdx {}/{}".format(trainExampleNdx, args.maximumNumberOfTrainingImages) )

numberOfValidationImages = min( int(args.maximumNumberOfTrainingImages * len(validationFileIDs)/len(trainFileIDs) ), len(validationFileIDs))
validationTensor = torch.Tensor(numberOfValidationImages, 3, args.imageSize, args.imageSize)
validationLabelTensor = torch.Tensor(numberOfValidationImages, numberOfBreeds).zero_()

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

    validationLabelTensor[validationExampleNdx, breedIndex] = 1.0

    if validationExampleNdx % 20 == 0:
        print("validationExampleNdx {}/{}".format(validationExampleNdx, numberOfValidationImages))


print ("validationLabelTensor = {}".format(validationLabelTensor))