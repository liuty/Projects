import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from OneMRecipeDataset import OneMRecipeDataset
import matplotlib.pyplot as plt

feature_maps = torch.load('feature_maps.pt')
labels = torch.load('labels.pt')

dataset1M_test = OneMRecipeDataset('test')
dataset1M_val = OneMRecipeDataset('val')
img_sample, lbl_sample = dataset1M_val[58]

vgg16 = models.vgg16(pretrained=True)
for param in vgg16.features.parameters():
	param.requires_grad = False

features = list(vgg16.classifier.children())[:-1]
vgg16.classifier = nn.Sequential(*features)
vgg16.cuda()
vgg16.eval()
fm_sample = vgg16(img_sample[None,:].cuda())

#breakpoint()
cossim = nn.CosineSimilarity(dim=1)
fm_sample = torch.cat(feature_maps.shape[0] * [fm_sample]) #tile the sample
diffs = cossim(fm_sample, feature_maps) #cosine similarity
values, indices = torch.topk(diffs, 5)
lbls = labels[indices].int() #remember these are from the test set

fig, ax = plt.subplots(1,len(lbls)+1)
ax[0].imshow(img_sample.permute(1,2,0))

for i in range(0,len(lbls)):
	img, idx = dataset1M_test[lbls[i]]
	ax[i+1].imshow(img.permute(1,2,0))

plt.show()
breakpoint()