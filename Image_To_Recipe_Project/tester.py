from OneMRecipeDataset import OneMRecipeDataset
from torchvision import datasets, models, transforms
from torch.autograd import Variable
import torch
import torch.nn as nn

def main():
	use_gpu = torch.cuda.is_available()

	dataset1M_train = OneMRecipeDataset('train')
	dataloader_train = torch.utils.data.DataLoader(dataset1M_train, batch_size=8, shuffle=True, num_workers=4)

	dataset1M_test = OneMRecipeDataset('test')
	dataloader_test = torch.utils.data.DataLoader(dataset1M_test, batch_size=8, shuffle=True, num_workers=4)

	dataset1M_val = OneMRecipeDataset('val')
	dataloader_val = torch.utils.data.DataLoader(dataset1M_val, batch_size=8, shuffle=True, num_workers=4)

	vgg16 = models.vgg16(pretrained=True)
	for param in vgg16.features.parameters():
		param.requires_grad = False

	features = list(vgg16.classifier.children())[:-1]
	vgg16.classifier = nn.Sequential(*features)
	if use_gpu:
		vgg16.cuda()

	feature_maps = torch.Tensor().cuda()
	ndxs = torch.Tensor().cuda()
	for i, data in enumerate(dataloader_test):
		vgg16.train(False)
		vgg16.eval()
		inputs, labels = data
		if use_gpu:
			inputs, labels = Variable(inputs.cuda(), volatile=True), Variable(labels.cuda(), volatile=True)
		outputs = vgg16(inputs)
		feature_maps = torch.cat((feature_maps, outputs), dim=0)
		ndxs = torch.cat((ndxs, labels), dim=0)

	torch.save(feature_maps, 'feature_maps.pt')
	torch.save(ndxs, 'labels.pt')
	breakpoint()


if __name__ == '__main__':
	torch.multiprocessing.freeze_support()
	main()