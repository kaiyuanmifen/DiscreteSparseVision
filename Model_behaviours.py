
import torch
import numpy as np
import torch.optim as optim 

from Dropout_DIY import *
from TaskModels import *


import matplotlib.pyplot as plt
import matplotlib

import torch
import torchvision
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np
import pandas as pd
import time
import h5py
from scipy.ndimage.interpolation import rotate
import sys
import argparse

matplotlib.use('AGG')





parser = argparse.ArgumentParser()


parser.add_argument('--seed', type=int, default=42,
					help='Random seed (default: 42).')


parser.add_argument('--Data', type=str, default='MNIST',
					help='Which data to use')

parser.add_argument('--Method', type=str, default='Original',
					help='dropout method')

parser.add_argument('--Epochs', type=int, default=50,
					help='Number of epochs')

parser.add_argument('--contrastive', type=str, default='no',
					help='whether do constrastive loss during training')

args = parser.parse_args()

args.contrastive=args.contrastive=='yes'

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
	torch.cuda.manual_seed(args.seed)


Task_name=args.Method+"_"+args.Data+"_"+str(args.contrastive)+"_"+str(args.seed)

print("task:",Task_name)







device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import os
directory="images/Examples/"
if not os.path.exists(directory):
    os.makedirs(directory)
####part 1 load MNIST/CIFAR data
batch_size=32
if args.Data=="MNIST":

	transform = transforms.Compose([transforms.ToTensor(), \
									transforms.Normalize((0), (1))])

	trainset = datasets.MNIST(root='../../data/', train=True, download=True, transform=transform)
	testset = datasets.MNIST(root='../../data/', train=False, transform=transform)

	indices = torch.randperm(len(trainset))[:1000]
	#indices = torch.randperm(len(trainset))

	trainset =torch.utils.data.Subset(trainset, indices)

	indices = torch.randperm(len(testset))[:1000]
	#indices = torch.randperm(len(testset))

	testset  =torch.utils.data.Subset(testset, indices)

	print("training set length")
	print(len(trainset))

	print("test set length")
	print(len(testset))

	# Visualize 10 image samples in MNIST dataset
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
	dataiter = iter(trainloader)
	images, labels = dataiter.next()


	# # plot 10 sample images
	_,ax = plt.subplots(1,10)
	ax = ax.flatten()
	iml = images[0].numpy().shape[1]
	[ax[i].imshow(np.transpose(images[i].numpy(),(1,2,0)).reshape(iml,-1),cmap='Greys') for i in range(10)]
	[ax[i].set_axis_off() for i in range(10)]
	plt.savefig('images/MNISTData.png')

	print('label:',labels[:10].numpy())
	print('image data shape:',images[0].numpy().shape)

	# #####augmented version


	#policies = [T.AutoAugmentPolicy.MNIST
	#half augmenters are used for GFN reward, the other half used for OOD testing
	augmenters=[transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 10))]
		

	#augmenter = transforms.RandomRotation(degrees=(0, 180))
	for idx,augmenter in enumerate(augmenters):
		#rotated_imgs = [rotater(orig_img) for _ in range(4)]
		Augmented_imgs=[augmenter(images[j]).unsqueeze(0) for j in range(10)]
		Augmented_imgs=torch.cat(Augmented_imgs,0)

		iml = Augmented_imgs[0].numpy().shape[1]
		[ax[i].imshow(np.transpose(Augmented_imgs[i].numpy(),(1,2,0)).reshape(iml,-1),cmap='Greys') for i in range(10)]
		[ax[i].set_axis_off() for i in range(10)]
		plt.savefig('images/AugmentedMNIST_'+str(idx)+'.png')


	# imgs = [
	# [augmenter(orig_img) for _ in range(4)]
	# for augmenter in augmenters
	# ]
if args.Data=="CIFAR10":

	transform = transforms.Compose(
	[transforms.ToTensor(),
	 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

	trainset = torchvision.datasets.CIFAR10(root='../../data', train=True,
											download=True, transform=transform)


	testset = torchvision.datasets.CIFAR10(root='../../data', train=False,
										   download=True, transform=transform)
	
	indices = torch.randperm(len(trainset))[:1000]
	#indices = torch.randperm(len(trainset))

	trainset =torch.utils.data.Subset(trainset, indices)

	indices = torch.randperm(len(testset))[:1000]
	#indices = torch.randperm(len(testset))

	testset  =torch.utils.data.Subset(testset, indices)
	
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
											  shuffle=True, num_workers=2)

	testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
											 shuffle=False, num_workers=2)

	print("training set length")
	print(len(trainset))

	print("test set length")
	print(len(testset))

	classes = ('plane', 'car', 'bird', 'cat',
			   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

	def imshow(img):
		img = img / 2 + 0.5     # unnormalize
		npimg = img.numpy()
		plt.imshow(np.transpose(npimg, (1, 2, 0)))
		plt.savefig('images/DataExamplesCIFAR.png')

	####show some examples
	dataiter = iter(trainloader)
	images, labels = dataiter.next()
	print("images")
	print(images.max())
	print(images.min())
	print(images.mean())
	
	print(images.shape)

	# show images
	imshow(torchvision.utils.make_grid(images))
	# print labels
	print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))
	
	#half augmenters are used for GFN reward, the other half used for OOD testing
	augmenters=[transforms.ColorJitter(brightness=.1,contrast=0.1,hue=0.05),
						transforms.ColorJitter(brightness=.2,contrast=0.2,hue=0.1),
						transforms.ColorJitter(brightness=.3,contrast=0.3,hue=0.15),
						transforms.ColorJitter(brightness=.4,contrast=0.4,hue=0.2),
						transforms.ColorJitter(brightness=.5,contrast=0.5,hue=0.25),
						transforms.ColorJitter(brightness=.6,contrast=0.6,hue=0.3),
						transforms.ColorJitter(brightness=.7,contrast=0.7,hue=0.35),
						transforms.ColorJitter(brightness=.8,contrast=0.8,hue=0.4),
						transforms.ColorJitter(brightness=.9,contrast=0.9,hue=0.45),]




	#augmenter = transforms.RandomRotation(degrees=(0, 180))
	for idx,augmenter in enumerate(augmenters):
		print("saving exmaples")
		#rotated_imgs = [rotater(orig_img) for _ in range(4)]
		Augmented_imgs=[augmenter(images[j]).unsqueeze(0) for j in range(30)]
		Augmented_imgs=torch.cat(Augmented_imgs,0)

		img=torchvision.utils.make_grid(Augmented_imgs)
		img = img / 2 + 0.5     # unnormalize
		npimg = img.numpy()
		plt.imshow(np.transpose(npimg, (1, 2, 0)))
		plt.savefig('images/Examples/AugmentedCIFAR_'+str(idx)+'.png')

if args.Data=="SVHN":

	transform = transforms.Compose(
	[transforms.ToTensor(),
	 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

	trainset = torchvision.datasets.SVHN(root='./data', split="train",
											download=True, transform=transform)


	testset = torchvision.datasets.SVHN(root='./data', split="test",
										   download=True, transform=transform)
	
	indices = torch.randperm(len(trainset))[:1000]
	#indices = torch.randperm(len(trainset))

	trainset =torch.utils.data.Subset(trainset, indices)

	indices = torch.randperm(len(testset))[:1000]
	#indices = torch.randperm(len(testset))

	testset  =torch.utils.data.Subset(testset, indices)
	
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
											  shuffle=True, num_workers=2)

	testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
											 shuffle=False, num_workers=2)

	print("training set length")
	print(len(trainset))

	print("test set length")
	print(len(testset))

	# classes = ('plane', 'car', 'bird', 'cat',
	# 		   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

	def imshow(img):
		img = img / 2 + 0.5     # unnormalize
		npimg = img.numpy()
		plt.imshow(np.transpose(npimg, (1, 2, 0)))
		plt.savefig('images/DataExamplesSVHN.png')

	####show some examples
	dataiter = iter(trainloader)
	images, labels = dataiter.next()
	print("images")
	print(images.shape)

	# show images
	imshow(torchvision.utils.make_grid(images))
	# print labels
	# print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))
	
	#half augmenters are used for GFN reward, the other half used for OOD testing
	augmenters=[transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 10))]
		

	#augmenter = transforms.RandomRotation(degrees=(0, 180))
	for idx,augmenter in enumerate(augmenters):
		#rotated_imgs = [rotater(orig_img) for _ in range(4)]
		Augmented_imgs=[augmenter(images[j]).unsqueeze(0) for j in range(30)]
		Augmented_imgs=torch.cat(Augmented_imgs,0)

		img=torchvision.utils.make_grid(Augmented_imgs)
		img = img / 2 + 0.5     # unnormalize
		npimg = img.numpy()
		plt.imshow(np.transpose(npimg, (1, 2, 0)))
		plt.savefig('images/AugmentedSVHN_'+str(idx)+'.png')




#part 2 function to run the task

class MLPClassifier:
	def __init__(self, image_size,droprates=0.5, batch_size=128, max_epoch=10, \
				 lr=3e-5, momentum=0,model_type="MLP_nodropout",N_units=50,
				 augmenter=None,name=None):
		# Wrap MLP model
		self.droprates = droprates
		self.batch_size = batch_size
		self.max_epoch = max_epoch
		self.model_type=model_type

		self.augmenters=augmenter
		self.name=name

		###pixels =>quantization => CNN=> MLP

		if self.model_type=="CNN_NoQuantization":
		  self.model = CNN_quantize(image_size=image_size,
		  					hidden_size=N_units,
		  					droprates=0,
		  					Method="Original")

		elif self.model_type=="CNN_HAQuantization":
			self.model = CNN_quantize(image_size=image_size,
					hidden_size=N_units,
					droprates=0,
					alpha=0.01,
					Method="Adaptive_Hierachical",
					N_factors=4)


		elif self.model_type=="CNN_HAQuantizationMask":
			self.model = CNN_quantize(image_size=image_size,
			hidden_size=N_units,
			droprates=0,
			alpha=0.01,
			Method="Adaptive_HierachicalMask",
			N_factors=4)


		elif self.model_type=="CNN_Quantization":
			self.model = CNN_quantize(image_size=image_size,
			hidden_size=N_units,
			droprates=0,
			alpha=0.01,
			Method="Quantization",
			N_factors=1)

		#### pixels => quantization => transformer
		elif self.model_type=="Transformer_NoQuantization":
			self.model = Transformer_quantize(image_size=image_size,
						output_dim=10,
        				Method="Original",alpha=0.01)

		elif self.model_type=="Transformer_HAQuantization":
			self.model = Transformer_quantize(image_size=image_size,
						output_dim=10,
        				Method="Adaptive_Hierachical",alpha=0.01,
						N_factors=4)

		elif self.model_type=="Transformer_HAQuantizationMask":
			self.model = Transformer_quantize(image_size=image_size,
						output_dim=10,
        				Method="Adaptive_HierachicalMask",alpha=0.01,
        				N_factors=4)

		elif self.model_type=="Transformer_Quantization":
			self.model = Transformer_quantize(image_size=image_size,
						output_dim=10,
        				Method="Quantization",alpha=0.01,
						N_factors=1)

		####CNN => quantization => MLP
		elif self.model_type=="CNNFeatures_NoQuantization":
			self.model = CNNFeature_quantize(image_size=image_size,
				output_dim=10,
				Method="Original",alpha=0.01)

		elif self.model_type=="CNNFeatures_HAQuantization":
			self.model = CNNFeature_quantize(image_size=image_size,
				output_dim=10,
				Method="Adaptive_Hierachical",alpha=0.01,
				N_factors=4)

		elif self.model_type=="CNNFeatures_HAQuantizationMask":
			self.model = CNNFeature_quantize(image_size=image_size,
				output_dim=10,
				Method="Adaptive_HierachicalMask",alpha=0.01,
				N_factors=4)

		elif self.model_type=="CNNFeatures_Quantization":
			self.model = CNNFeature_quantize(image_size=image_size,
			output_dim=10,
			Method="Quantization",alpha=0.01,
			N_factors=1)
			
		#####CNN feature => quantization => transformer

		elif self.model_type=="Hybrid_NoQuantization":
			self.model = CNNTransformerMixture_quantize(image_size=image_size,
						output_dim=10,
        				Method="Original",alpha=0.01)

		elif self.model_type=="Hybrid_HAQuantization":
			self.model = CNNTransformerMixture_quantize(image_size=image_size,
						output_dim=10,
        				Method="Adaptive_Hierachical",alpha=0.01,
						N_factors=4)

		elif self.model_type=="Hybrid_HAQuantizationMask":
			self.model = CNNTransformerMixture_quantize(image_size=image_size,
						output_dim=10,
        				Method="Adaptive_HierachicalMask",alpha=0.01,
						N_factors=4)


		elif self.model_type=="Hybrid_Quantization":
			self.model = CNNTransformerMixture_quantize(image_size=image_size,
						output_dim=10,
        				Method="Quantization",alpha=0.01,
						N_factors=1)




		self.model.to(device)
		self.criterion = nn.CrossEntropyLoss().to(device)
		



		####load the model
		PATH="checkpoints/"+Task_name+".pt"
		try:
			
			file = open(PATH, 'r')
			print("loading model from: ",file)
			self.model.load_state_dict(torch.load(PATH,map_location=device))
			self.model.eval()
		except IOError:
			print('no model file not existing')
			sys.exit()


		self.model.to(device)
		

		self.criterion = nn.CrossEntropyLoss().to(device)
		
		self.loss_ = []
		self.Quantization_losses=[]
		self.test_accuracy = []
		self.Qatt_scores=[]


	
		total_params = sum(p.numel() for p in self.model.parameters())


		print("number of parameters:")
		print(total_params)


	def test(self, testset):


		self.model.eval()
		testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False)
		X_test, y_test = iter(testloader).next()
		X_test = X_test.to(device)
		y_test=y_test.to(device)
		

		for idx,augmenter_ in enumerate(self.augmenters):

			with torch.no_grad():
				
				if idx>0:
					X_test_=augmenter_(X_test)
				else:
					X_test_=X_test
				outputs,Quantization_loss,att_scores,constrastive_loss = self.model(X_test_)
				
				loss = self.criterion(outputs, y_test)

				_, pred = torch.max(outputs.data, 1)
				
				y_test_pred=pred
				acc=np.mean((y_test == y_test_pred).cpu().numpy())


			self.loss_.append(loss.item())
			self.Quantization_losses.append(Quantization_loss.item())
			self.test_accuracy.append(acc)
			self.Qatt_scores.append(att_scores.mean(1).squeeze(0).tolist())

				
		df = pd.DataFrame({'test_loss':self.loss_,
						'test_acc':self.test_accuracy,
						"Quantization_losses":self.Quantization_losses,
						"Qatt_scores":self.Qatt_scores
							})
				

		import os
		if not os.path.exists("Results/BehaviourAnalysis/"):
			os.makedirs("Results/BehaviourAnalysis/")
		df.to_csv("Results/BehaviourAnalysis/"+Task_name+"_"+self.name+"_BehaviourAnalysis.csv")

	
	def predict(self, x):
		# Used to keep all test errors after each epoch
		model = self.model.eval()
		with torch.no_grad():
			
			outputs,_,_,_ = model(Variable(x))

			_, pred = torch.max(outputs.data, 1)
		model = self.model.train()
		return pred
	
	def __str__(self):
		return 'Hidden layers: {}; dropout rates: {}'.format(self.hidden_layers, self.droprates)




#######run the model 

## Below is training code, uncomment to train your own model... ###
### Note: You need GPU to run this section ###

# Define networks

if args.Data=="MNIST":
	#Input_size=28*28
	image_size=(1,28,28)
elif args.Data=="CIFAR10":
	#Input_size=3*32*32
	image_size=(3,32,32)
elif args.Data=="SVHN":
	#Input_size=3*32*32
	image_size=(3,32,32)

#####different augmenters
####Gaussian
Gaussian_augmenters=[[],
	transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 0.1)),
	transforms.GaussianBlur(kernel_size=(5, 9), sigma=(1.0, 1.0)),
	transforms.GaussianBlur(kernel_size=(5, 9), sigma=(2.0, 2.0)),
	transforms.GaussianBlur(kernel_size=(5, 9), sigma=(5.0, 5.0)),
	transforms.GaussianBlur(kernel_size=(5, 9), sigma=(10.0, 10.0)),
	transforms.GaussianBlur(kernel_size=(5, 9), sigma=(15.0, 15.0)),
	transforms.GaussianBlur(kernel_size=(5, 9), sigma=(20.0, 20.0)),
	transforms.GaussianBlur(kernel_size=(5, 9), sigma=(30.0, 30.0)),
	transforms.GaussianBlur(kernel_size=(5, 9), sigma=(35.0, 35.0)),
	transforms.GaussianBlur(kernel_size=(5, 9), sigma=(40.0, 40.0))]



Rotations_augmenters=[transforms.RandomRotation(degrees=(0,10)),
						transforms.RandomRotation(degrees=(10,20)),
						transforms.RandomRotation(degrees=(20,30)),
						transforms.RandomRotation(degrees=(30,40)),
						transforms.RandomRotation(degrees=(40,50)),
						transforms.RandomRotation(degrees=(50,60)),
						transforms.RandomRotation(degrees=(60,70)),
						transforms.RandomRotation(degrees=(70,80)),
						transforms.RandomRotation(degrees=(80,90)),]



Solarize_augmenters=[transforms.RandomSolarize(threshold=192.0),
						transforms.RandomSolarize(threshold=162.0),
						transforms.RandomSolarize(threshold=142.0),
						transforms.RandomSolarize(threshold=122.0),
						transforms.RandomSolarize(threshold=102.0),
						transforms.RandomSolarize(threshold=82.0),
						transforms.RandomSolarize(threshold=62.0),
						transforms.RandomSolarize(threshold=42.0),
						transforms.RandomSolarize(threshold=22.0),]


colorjitter_augmenters=[transforms.ColorJitter(brightness=.1,contrast=0.1,hue=0.05),
						transforms.ColorJitter(brightness=.2,contrast=0.2,hue=0.1),
						transforms.ColorJitter(brightness=.3,contrast=0.3,hue=0.15),
						transforms.ColorJitter(brightness=.4,contrast=0.4,hue=0.2),
						transforms.ColorJitter(brightness=.5,contrast=0.5,hue=0.25),
						transforms.ColorJitter(brightness=.6,contrast=0.6,hue=0.3),
						transforms.ColorJitter(brightness=.7,contrast=0.7,hue=0.35),
						transforms.ColorJitter(brightness=.8,contrast=0.8,hue=0.4),
						transforms.ColorJitter(brightness=.9,contrast=0.9,hue=0.45),]





if args.Data=="MNIST":
	Random_augmenters=[transforms.AutoAugmentPolicy.MNIST]
elif args.Data=="CIFAR10":
	Random_augmenters=[transforms.AutoAugmentPolicy.CIFAR10]
elif args.Data=="SVHN":
	Random_augmenters=[transforms.AutoAugmentPolicy.SVHN]



mlp1 = [MLPClassifier(image_size=image_size,max_epoch=args.Epochs,model_type=args.Method,N_units=200,augmenter=colorjitter_augmenters,name="Colorjitter"),
		MLPClassifier(image_size=image_size,max_epoch=args.Epochs,model_type=args.Method,N_units=200,augmenter=Solarize_augmenters,name="Solarize"),
		MLPClassifier(image_size=image_size,max_epoch=args.Epochs,model_type=args.Method,N_units=200,augmenter=Gaussian_augmenters,name="Gaussian")]

#MLPClassifier(image_size=image_size,max_epoch=args.Epochs,model_type=args.Method,N_units=200,augmenter=Solarize_augmenters,name="Solarize")]

# mlp1 = [MLPClassifier(image_size=image_size,max_epoch=args.Epochs,model_type=args.Method,N_units=200,augmenter=Rotations_augmenters,name="Rotation"),
# 		MLPClassifier(image_size=image_size,max_epoch=args.Epochs,model_type=args.Method,N_units=200,augmenter=Gaussian_augmenters,name="Gaussian"),
# 		MLPClassifier(image_size=image_size,max_epoch=args.Epochs,model_type=args.Method,N_units=200,augmenter=Random_augmenters,name="Random")]
[mlp.test(testset) for mlp in mlp1]
