
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

import argparse
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader
from wilds.common.data_loaders import get_eval_loader
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


####part 1 load MNIST/CIFAR data
batch_size=32


import os
directory="images/Examples/"
if not os.path.exists(directory):
    os.makedirs(directory)
if args.Data=="MNIST":

	transform = transforms.Compose([transforms.ToTensor(), \
									transforms.Normalize((0), (1))])

	trainset = datasets.MNIST(root='../../data/', train=True, download=True, transform=transform)
	testset = datasets.MNIST(root='../../data/', train=False, transform=transform)

	#indices = torch.randperm(len(trainset))[:1000]
	indices = torch.randperm(len(trainset))

	trainset =torch.utils.data.Subset(trainset, indices)

	#indices = torch.randperm(len(testset))[:300]
	indices = torch.randperm(len(testset))

	testset  =torch.utils.data.Subset(testset, indices)

	print("training set length")
	print(len(trainset))

	print("test set length")
	print(len(testset))

	# Visualize 10 image samples in MNIST dataset
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
	testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
	

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
	augmenters=[transforms.GaussianBlur(kernel_size=(5, 9), sigma=(5, 5))]
		

	#augmenter = transforms.RandomRotation(degrees=(0, 180))
	for idx,augmenter in enumerate(augmenters):
		#rotated_imgs = [rotater(orig_img) for _ in range(4)]
		Augmented_imgs=[augmenter(images[j]).unsqueeze(0) for j in range(10)]
		Augmented_imgs=torch.cat(Augmented_imgs,0)

		iml = Augmented_imgs[0].numpy().shape[1]
		[ax[i].imshow(np.transpose(Augmented_imgs[i].numpy(),(1,2,0)).reshape(iml,-1),cmap='Greys') for i in range(10)]
		[ax[i].set_axis_off() for i in range(10)]
		plt.savefig('AugmentedMNIST_'+str(idx)+'.png')


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
	
	#indices = torch.randperm(len(trainset))[:6000]
	indices = torch.randperm(len(trainset))

	trainset =torch.utils.data.Subset(trainset, indices)

	#indices = torch.randperm(len(testset))[:300]
	indices = torch.randperm(len(testset))

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
	print(images.shape)

	# show images
	imshow(torchvision.utils.make_grid(images))
	# print labels
	print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))
	
	#half augmenters are used for GFN reward, the other half used for OOD testing
	augmenters=[transforms.GaussianBlur(kernel_size=(5, 9), sigma=(5, 5))]
		

	#augmenter = transforms.RandomRotation(degrees=(0, 180))
	for idx,augmenter in enumerate(augmenters):
		#rotated_imgs = [rotater(orig_img) for _ in range(4)]
		Augmented_imgs=[augmenter(images[j]).unsqueeze(0) for j in range(30)]
		Augmented_imgs=torch.cat(Augmented_imgs,0)

		img=torchvision.utils.make_grid(Augmented_imgs)
		img = img / 2 + 0.5     # unnormalize
		npimg = img.numpy()
		plt.imshow(np.transpose(npimg, (1, 2, 0)))
		plt.savefig('AugmentedCIFAR_'+str(idx)+'.png')

if args.Data=="SVHN":

	transform = transforms.Compose(
	[transforms.ToTensor(),
	 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

	trainset = torchvision.datasets.SVHN(root='../../data', split="train",
											download=True, transform=transform)


	testset = torchvision.datasets.SVHN(root='../../data', split="test",
										   download=True, transform=transform)
	
	#indices = torch.randperm(len(trainset))[:6000]
	indices = torch.randperm(len(trainset))

	trainset =torch.utils.data.Subset(trainset, indices)

	#indices = torch.randperm(len(testset))[:10000]
	indices = torch.randperm(len(testset))

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
	augmenters=[transforms.GaussianBlur(kernel_size=(5, 9), sigma=(5, 5))]
		

	#augmenter = transforms.RandomRotation(degrees=(0, 180))
	for idx,augmenter in enumerate(augmenters):
		#rotated_imgs = [rotater(orig_img) for _ in range(4)]
		Augmented_imgs=[augmenter(images[j]).unsqueeze(0) for j in range(30)]
		Augmented_imgs=torch.cat(Augmented_imgs,0)

		img=torchvision.utils.make_grid(Augmented_imgs)
		img = img / 2 + 0.5     # unnormalize
		npimg = img.numpy()
		plt.imshow(np.transpose(npimg, (1, 2, 0)))
		plt.savefig('AugmentedSVHN_'+str(idx)+'.png')

if args.Data== 'camelyon17':
	transform = transforms.Compose(
	[transforms.Resize((96, 96)),transforms.ToTensor(),
	 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

	dataset = get_dataset(dataset= 'camelyon17', download=True,root_dir='../../data')

	trainset = dataset.get_subset(
	"train",transform=transform,frac=0.01)

	testset = dataset.get_subset(
	"test",transform=transform,frac=0.1)




	trainloader = get_train_loader("standard", trainset, batch_size=batch_size)

	testloader = get_eval_loader("standard", testset, batch_size=batch_size)


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
		plt.savefig('images/DataExamplesCamelyon17.png')

	####show some examples
	dataiter = iter(trainloader)
	images, labels ,metadata= dataiter.next()
	print("images")
	print(images.shape)
	print("labels")
	print(labels)

	# show images
	imshow(torchvision.utils.make_grid(images))
	# print labels
	# print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))
	
	#half augmenters are used for GFN reward, the other half used for OOD testing
	augmenters=[transforms.GaussianBlur(kernel_size=(5, 9), sigma=(5, 5))]
		

	#augmenter = transforms.RandomRotation(degrees=(0, 180))
	for idx,augmenter in enumerate(augmenters):
		#rotated_imgs = [rotater(orig_img) for _ in range(4)]
		Augmented_imgs=[augmenter(images[j]).unsqueeze(0) for j in range(30)]
		Augmented_imgs=torch.cat(Augmented_imgs,0)

		img=torchvision.utils.make_grid(Augmented_imgs)
		img = img / 2 + 0.5     # unnormalize
		npimg = img.numpy()
		plt.imshow(np.transpose(npimg, (1, 2, 0)))
		plt.savefig('AugmentedCamelyon17_'+str(idx)+'.png')


#part 2 function to run the task

class MLPClassifier:
	def __init__(self, image_size,droprates=0.5, batch_size=128, max_epoch=10, \
				 lr=3e-5, momentum=0,model_type=None,N_units=50,alpha=0.1):
		# Wrap MLP model
		self.droprates = droprates
		self.batch_size = batch_size
		self.max_epoch = max_epoch
		self.model_type=model_type

		if args.Data in ["CIFAR10","MNIST","SVHN"]:
			output_dim=10
		elif args.Data in ['camelyon17']:
			output_dim=2

		###pixels =>quantization => CNN=> MLP

		if self.model_type=="CNN_NoQuantization":
		  self.model = CNN_quantize(image_size=image_size,
		  					hidden_size=N_units,
		  					droprates=0,
		  					Method="Original",output_dim=output_dim)

		elif self.model_type=="CNN_HAQuantization":
			self.model = CNN_quantize(image_size=image_size,
					hidden_size=N_units,
					droprates=0,
					alpha=alpha,
					Method="Adaptive_Hierachical",
					N_factors=4,output_dim=output_dim)


		elif self.model_type=="CNN_HAQuantizationMask":
			self.model = CNN_quantize(image_size=image_size,
			hidden_size=N_units,
			droprates=0,
			alpha=alpha,
			Method="Adaptive_HierachicalMask",
			N_factors=4,output_dim=output_dim)


		elif self.model_type=="CNN_Quantization":
			self.model = CNN_quantize(image_size=image_size,
			hidden_size=N_units,
			droprates=0,
			alpha=alpha,
			Method="Quantization",
			N_factors=1,output_dim=output_dim)

		#### pixels => quantization => transformer
		elif self.model_type=="Transformer_NoQuantization":
			self.model = Transformer_quantize(image_size=image_size,
        				Method="Original",alpha=alpha,output_dim=output_dim)

		elif self.model_type=="Transformer_HAQuantization":
			self.model = Transformer_quantize(image_size=image_size,
        				Method="Adaptive_Hierachical",alpha=alpha,
						N_factors=4,output_dim=output_dim)

		elif self.model_type=="Transformer_HAQuantizationMask":
			self.model = Transformer_quantize(image_size=image_size,
        				Method="Adaptive_HierachicalMask",alpha=alpha,
        				N_factors=4,output_dim=output_dim)

		elif self.model_type=="Transformer_Quantization":
			self.model = Transformer_quantize(image_size=image_size,
        				Method="Quantization",alpha=alpha,
						N_factors=1,output_dim=output_dim)

		####CNN => quantization => MLP
		elif self.model_type=="CNNFeatures_NoQuantization":
			self.model = CNNFeature_quantize(image_size=image_size,
				Method="Original",alpha=alpha,output_dim=output_dim)

		elif self.model_type=="CNNFeatures_HAQuantization":
			self.model = CNNFeature_quantize(image_size=image_size,
				Method="Adaptive_Hierachical",alpha=alpha,
				N_factors=4,output_dim=output_dim)

		elif self.model_type=="CNNFeatures_HAQuantizationMask":
			self.model = CNNFeature_quantize(image_size=image_size,
				Method="Adaptive_HierachicalMask",alpha=alpha,
				N_factors=4,output_dim=output_dim)

		elif self.model_type=="CNNFeatures_Quantization":
			self.model = CNNFeature_quantize(image_size=image_size,
			Method="Quantization",alpha=alpha,
			N_factors=1,output_dim=output_dim)
			
		#####CNN feature => quantization => transformer

		elif self.model_type=="Hybrid_NoQuantization":
			self.model = CNNTransformerMixture_quantize(image_size=image_size,
        				Method="Original",alpha=alpha,output_dim=output_dim)

		elif self.model_type=="Hybrid_HAQuantization":
			self.model = CNNTransformerMixture_quantize(image_size=image_size,
        				Method="Adaptive_Hierachical",alpha=alpha,
						N_factors=4,output_dim=output_dim)

		elif self.model_type=="Hybrid_HAQuantizationMask":
			self.model = CNNTransformerMixture_quantize(image_size=image_size,
        				Method="Adaptive_HierachicalMask",alpha=alpha,
						N_factors=4,output_dim=output_dim)


		elif self.model_type=="Hybrid_Quantization":
			self.model = CNNTransformerMixture_quantize(image_size=image_size,
        				Method="Quantization",alpha=alpha,
						N_factors=1,output_dim=output_dim)




		self.model.to(device)
		self.criterion = nn.CrossEntropyLoss().to(device)
		
		###as this is a transformer training
		self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
		gamma = 0.7
		from torch.optim.lr_scheduler import StepLR
		self.scheduler = StepLR(self.optimizer, step_size=1, gamma=gamma)
		

		self.loss_ = []
		self.task_losses = []
		self.Quantization_losses=[]
		self.Contrastive_losses=[]
		self.test_accuracy = []
		self.test_error = []
		self.test_accuracy_OOD = []
		self.test_error_OOD = []
		self.test_losses = []
		self.OODtest_losses = []
	
		total_params = sum(p.numel() for p in self.model.parameters())


		print("number of parameters:")
		print(total_params)


	def fit(self, trainset, testset, verbose=True):
		# Training, make sure it's on GPU, otherwise, very slow...
		if args.Data== 'camelyon17':
			X_test, y_test,metadata = iter(testloader).next()
		else:
			X_test, y_test = iter(testloader).next()
		X_test = X_test.to(device)

		best_test_acc=0

		

		for epoch in range(self.max_epoch):
			
			running_loss = 0
			running_task_loss=0
			running_Qloss=0
			running_Closs=0
			for i, data in enumerate(trainloader, 0):
				if args.Data== 'camelyon17':
					inputs_, labels_,metadata = data

				else:
					inputs_, labels_ = data

				inputs, labels = Variable(inputs_).to(device), Variable(labels_).to(device)
				self.optimizer.zero_grad()

	
			   	###forward
			
				outputs,Quantization_loss,att_scores,contrastive_loss = self.model(inputs)

				loss = self.criterion(outputs, labels)
				running_task_loss+=loss.item()
				if args.contrastive:
					#print("using constrastive loss")
					loss=loss+Quantization_loss+contrastive_loss
				else:
					#print("not using constrastive loss")
					loss=loss+Quantization_loss
				loss.backward()
				self.optimizer.step()
				running_loss += loss.item()
				running_Qloss+=Quantization_loss.item()
				running_Closs+=contrastive_loss.item()

				

			self.loss_.append(running_loss / len(trainloader))
			self.Quantization_losses.append(running_Qloss/len(trainloader))
			self.Contrastive_losses.append(running_Closs/len(trainloader))
			self.task_losses.append(running_task_loss/len(trainloader))

			if verbose and epoch%1==0:
				print('Epoch {} loss: {} QLoss: {} ContrLoss {}'.format(epoch+1, self.loss_[-1],self.Quantization_losses[-1],self.Contrastive_losses[-1]))
			
  


			acc,test_loss= self.predict()
			self.test_accuracy.append(acc)
			self.test_error.append(int(len(testset)*(1-self.test_accuracy[-1])))
			self.test_losses.append(test_loss)
			####OOD loss
			OOD_accs=[]
			OOD_testerrors=[]
			OODtest_loss=[]
			for idx,augmenter in enumerate(augmenters):
				acc,OOD_loss= self.predict(augmenter=augmenter)
				OOD_accs.append(acc)
				OOD_testerrors.append((int(len(testset)*(1-acc))))
				OODtest_loss.append(OOD_loss)

			
			self.test_accuracy_OOD.append(np.mean(OOD_accs))
			self.test_error_OOD.append(np.mean(OOD_testerrors))
			self.OODtest_losses.append(np.mean(OODtest_loss))

			#df = pd.DataFrame({'train_loss':self.loss_,'test_acc':self.test_accuracy,'test_error':self.test_error,"GFN_loss":self.GFN_loss})
			
			df = pd.DataFrame({'Total_train_loss':self.loss_,
							'task_losses':self.task_losses,
							'test_acc':self.test_accuracy,
							'test_error':self.test_error,
							"test_loss":self.test_losses,
							"Quantization_losses":self.Quantization_losses,
							'test_acc_OOD':self.test_accuracy_OOD,
							'test_error_OOD':self.test_error_OOD,
							"test_loss_OOD":self.OODtest_losses,
							"constrastive_loss":self.Contrastive_losses})
					


			df.to_csv("Results/"+Task_name+"_performance.csv")

			#if best validation loss save model
			if acc>best_test_acc:
				torch.save(self.model.state_dict(), "checkpoints/"+Task_name+'.pt')
				best_test_acc=acc

			if verbose and epoch%1==0:
				print('Test error: {}; test accuracy: {}'.format(self.test_error[-1], self.test_accuracy[-1]))
		return self
	
	def predict(self, augmenter=None):
		# Used to keep all test errors after each epoch
		model = self.model.eval()
		
		acc=[]
		with torch.no_grad():
			test_loss=0

			for i, data in enumerate(testloader, 0):
				if args.Data== 'camelyon17':
					X_test, Y_test,metadata = data

				else:
					X_test, Y_test = data
			
				X_test=X_test.to(device)
				Y_test=Y_test.to(device)
				if augmenter!=None:
					X_test=augmenter(X_test.detach().clone().to(device))
				
				outputs,_,_,_ = model(Variable(X_test.to(device)))

				_, Y_test_pred = torch.max(outputs.data, 1)

				batch_acc=np.mean((Y_test == Y_test_pred).cpu().numpy())
				acc.append(batch_acc)

				batch_loss = self.criterion(outputs, Y_test)
				test_loss+=batch_loss.item()

			acc=np.mean(acc)
			test_loss=test_loss/len(testloader)

		model = self.model.train()
		return acc,test_loss
	
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
elif args.Data== 'camelyon17':
	image_size=(3,96,96)

mlp1 = [MLPClassifier(image_size=image_size,max_epoch=args.Epochs,model_type=args.Method,N_units=200)]

# mlp1 = [MLPClassifier(droprates=[0.0, 0.5], max_epoch=3,model_type="MLP"),
#         MLPClassifier(droprates=[0.0, 0.0], max_epoch=3,model_type="MLP_GFN"),
#         MLPClassifier(droprates=[0.0, 0.0], max_epoch=3,model_type="MLP_SVD"),
#        MLPClassifier(droprates=[0.0, 0.0], max_epoch=3,model_type="MLP_Standout")
#         ]
#       
# Training, set verbose=True to see loss after each epoch.
[mlp.fit(trainset, testset,verbose=True) for mlp in mlp1]

# Save torch models
# for ind, mlp in enumerate(mlp1):
# 	#torch.save(mlp.model, 'mnist_mlp1_'+str(ind)+'.pth')
# 	#torch.save(mlp.model, "checkpoints/"+Task_name+'.pth')
# 	torch.save(mlp.model.state_dict(), "checkpoints/"+Task_name+'.pt')
	
	# Prepare to save errors
	#mlp.test_error = list(map(str, mlp.test_error))

# Save test errors to plot figures
#open("Results/"+Task_name+"test_errors.txt","w").write('\n'.join([','.join(mlp.test_error) for mlp in mlp1])) 


# Load saved models to CPU
#mlp1_models = [torch.load('mnist_mlp1_'+str(ind)+'.pth',map_location={'cuda:0': 'cpu'}) for ind in [0,1,2]]

# Load saved test errors to plot figures.
# mlp1_test_errors = [error_array.split(',') for error_array in open("Results/"+Task_name+"test_errors.txt","r").read().split('\n')]
# mlp1_test_errors = np.array(mlp1_test_errors,dtype='f')




#####visualization 


# labels = [args.Method] 
# #          'MLP 50% dropout in hidden layers',
#  #         'MLP 50% dropout in hidden layers+20% input layer']

# plt.figure(figsize=(8, 7))
# for i, r in enumerate(mlp1_test_errors):
# 	plt.plot(range(1, len(r)+1), r, '.-', label=labels[i], alpha=0.6);
# #plt.ylim([50, 250]);
# plt.legend(loc=1);
# plt.xlabel('Epochs');
# plt.ylabel('Number of errors in test set');
# plt.title('Test error on MNIST dataset for Multilayer Perceptron')
# plt.savefig('Results/'+Task_name+'PerformanceVsepisodes.png')
# plt.clf()
