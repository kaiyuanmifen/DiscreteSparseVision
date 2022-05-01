
import torch
import numpy as np

import torch.nn as nn

import torch.nn.functional as F

import torch

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torch.nn import Parameter
from Dropout_DIY import *


from Quantization import *

from QuantizerFunction import QuantizerFunction


from vit_pytorch import ViT



class CNN_quantize(nn.Module):
    def __init__(self,image_size,hidden_size=10,droprates=0,
                Method="Original",N_factors=1,alpha=1.0,output_dim=10):
        super(CNN_quantize, self).__init__()
        self.conv1 = nn.Conv2d(image_size[0], 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32,64, kernel_size=5)

        self.image_size=image_size
        if self.image_size[1]==28:
            self.CNNoutputsize=3*3*64
        elif self.image_size[1]==32:
            self.CNNoutputsize=4*4*64


        self.fc1 = nn.Linear(self.CNNoutputsize, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_dim)

        self.droprates=droprates

      
        #####quantization code 


        self.quantize=QuantizerFunction(input_dims=64,
                                        CodebookSize=128,
                                        Method=Method,
                                        N_factor=N_factors,
                                        alpha=alpha,
                                        N_discretizers=3)


    def forward(self, x):

        #x=F.dropout(x, p=self.droprates[0], training=self.training)
        x = F.relu(self.conv1(x))
        #x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        #x = F.dropout(x, p=self.droprates[1], training=self.training)
        x = F.relu(F.max_pool2d(self.conv3(x),2))
        #x = F.dropout(x, p=self.droprates[1], training=self.training)
        #conduct quantizaiton on the CNN output 
 
        bsz,n_channels,H,W=x.shape
        x,ExtraLoss,att_scores=self.quantize(x.permute(0,2,3,1).reshape(bsz*H*W,1,n_channels))
        
     
        x = x.view(-1,self.CNNoutputsize)
        x = F.relu(self.fc1(x))
        #x = F.dropout(x, p=self.droprates,training=self.training)
        x = self.fc2(x)
        return x,ExtraLoss,att_scores


#######transformer based model 
from vit_pytorch import ViT
###directly use ViT code to avoid problem

class Transformer_quantize(nn.Module):
    """Transformer on image model"""
    
    def __init__(self, image_size,output_dim,hidden_dim=2048,
        Method="Original",alpha=1.0,N_factors=1):
        super(Transformer_quantize, self).__init__()

     

        self.patch_size=4
        self.transformer=ViT( image_size = image_size[1],###this actually mean image height
                                patch_size = self.patch_size,
                                num_classes = output_dim,
                                dim = 128,
                                depth = 6,
                                heads = 4,
                                mlp_dim = hidden_dim,
                                dropout = 0.1,
                                emb_dropout = 0.1
                            )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        #####quantization code 


        self.quantize=QuantizerFunction(input_dims=3*self.patch_size*self.patch_size,
                        CodebookSize=128,
                        Method=Method,
                        N_factor=N_factors,
                        alpha=alpha,
                        N_discretizers=3)

        self.to(self.device)
 

    def forward(self,s_t):
        
        '''

        s_t has shape (bsz,n_channels,H,W)
        '''
        bsz,n_channels,H,W=s_t.shape
        s_t=s_t.reshape(bsz,-1,n_channels*self.patch_size*self.patch_size)
        bsz,T,dim=s_t.shape
        
        ####conduct quantization
        s_t=s_t.reshape(T*bsz,1,dim)
        s_t,ExtraLoss,att_scores=self.quantize(s_t)
        s_t=s_t.reshape(bsz,T,dim).reshape(bsz,n_channels,H,W)
        
        x=self.transformer(s_t)


        return x,ExtraLoss,att_scores
 





class CNNEncoder(nn.Module):
    def __init__(self,image_size,droprates=0):
        super(CNNEncoder, self).__init__()
        self.conv1 = nn.Conv2d(image_size[0], 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32,16, kernel_size=5)

        self.image_size=image_size
        if self.image_size[1]==28:
            self.CNNoutputsize=3*3*64
        elif self.image_size[1]==32:
            self.CNNoutputsize=4*4*64


    def forward(self, x):

        #x=F.dropout(x, p=self.droprates[0], training=self.training)
        x = F.relu(self.conv1(x))
        #x = F.dropout(x, p=0.5, training=self.training)
        #x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(self.conv2(x))
        
        #x = F.dropout(x, p=self.droprates[1], training=self.training)
        #x = F.relu(F.max_pool2d(self.conv3(x),2))
        x = F.relu(self.conv3(x))
        #x = F.dropout(x, p=self.droprates[1], training=self.training)

        return x

class CNNDecoder(nn.Module):
    def __init__(self,Input_N_channels):
        super().__init__()

        self.decoder = nn.Sequential(
        nn.ConvTranspose2d(Input_N_channels, 16, 6, stride=1),  
        nn.ReLU(True),
        nn.ConvTranspose2d(16, 8, 7, stride=1, padding=1),  
        nn.ReLU(True),
        nn.ConvTranspose2d(8, 3, 6, stride=1, padding=1), 
        nn.Tanh()
        )
    def forward(self, x):
        x=self.decoder(x)
        return x




#################CNN=> quantizaiton => MLP


class CNN_quantize(nn.Module):
    def __init__(self,image_size,hidden_size=10,droprates=0,output_dim=10,
                Method="Original",alpha=1.0,N_factors=1):
        super(CNN_quantize, self).__init__()
        self.conv1 = nn.Conv2d(image_size[0], 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32,16, kernel_size=5)

        self.image_size=image_size

        if self.image_size==(3,32,32):
            MLP_Input_dim=16*16
        elif self.image_size==(1,28,28):
            MLP_Input_dim=16*9
        elif self.image_size==(3,96,96):
            MLP_Input_dim=16*20*20

        self.fc1 = nn.Linear(MLP_Input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_dim)

        self.droprates=droprates

      
        #####quantization code 
        self.patch_size=4

        self.quantize=QuantizerFunction(input_dims=self.patch_size*self.patch_size,
                                        CodebookSize=128,
                                        Method=Method,
                                        N_factor=N_factors,
                                        alpha=alpha,
                                        N_discretizers=3)


    def forward(self, x):

        ###quantization first
        bsz,n_channels,H,W=x.shape
        x=x.reshape(bsz*n_channels,1,H,W)
        x,ExtraLoss,att_scores=self.quantize(x.reshape(-1,1,self.patch_size*self.patch_size))
        x=x.reshape(bsz*n_channels,1,H,W).reshape(bsz,n_channels,H,W)

        #x=F.dropout(x, p=self.droprates[0], training=self.training)
        x = F.relu(self.conv1(x))
        #x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        #x = F.dropout(x, p=self.droprates[1], training=self.training)
        x = F.relu(F.max_pool2d(self.conv3(x),2))
        #x = F.dropout(x, p=self.droprates[1], training=self.training)
 
        x = x.view(bsz,-1)
        x = F.relu(self.fc1(x))
        #x = F.dropout(x, p=self.droprates,training=self.training)
        Pred = self.fc2(x)

        contrativeLoss=torch.zeros(1)#placeholder

        return Pred,ExtraLoss,att_scores,contrativeLoss


#######image => quantization=> transformer


class Transformer_quantize(nn.Module):
    def __init__(self,image_size,output_dim,hidden_dim=2048,
                Method="Original",N_factors=1,alpha=1.0):
        super(Transformer_quantize, self).__init__()


        self.patch_size=4
        
        self.transformer=ViT( image_size = image_size[1],###this actually mean image height
                                channels = image_size[0],
                                patch_size = self.patch_size,
                                num_classes = output_dim,
                                dim = 128,
                                depth = 6,
                                heads = 4,
                                mlp_dim = hidden_dim,
                                dropout = 0.1,
                                emb_dropout = 0.1
                            )

      
        #####quantization code 


        self.quantize=QuantizerFunction(input_dims=self.patch_size*self.patch_size,
                                        CodebookSize=128,
                                        Method=Method,
                                        N_factor=N_factors,
                                        alpha=alpha,
                                        N_discretizers=3)


    def forward(self, x):

        ###quantization first
        bsz,n_channels,H,W=x.shape
        x=x.reshape(bsz*n_channels,1,H,W)
        x,ExtraLoss,att_scores=self.quantize(x.reshape(-1,1,self.patch_size*self.patch_size))
        x=x.reshape(bsz*n_channels,1,H,W).reshape(bsz,n_channels,H,W)

        Pred=self.transformer(x)

      
        contrativeLoss=torch.zeros(1)#placeholder

        return Pred,ExtraLoss,att_scores,contrativeLoss



######CNN representation learning 
class CNNFeature_quantize(nn.Module):
    """CNN as feature learner"""
    
    def __init__(self, image_size,output_dim,hidden_dim=2256,
        Method="Original",alpha=0.01,N_factors=1):
        super(CNNFeature_quantize, self).__init__()

        

        self.patch_size=4
        self.image_size=image_size

        #####CNNPart
        self.CNNencoder =CNNEncoder(image_size=image_size)

        self.CNNdecoder=CNNDecoder(Input_N_channels=16)

        #####MLP part 
        if self.image_size==(3,32,32):
            MLP_Input_dim=16*20*20
        elif self.image_size==(1,28,28):
            MLP_Input_dim=16*16*16
        elif self.image_size==(3,96,96):
            MLP_Input_dim=16*20*20
        self.fc1 = nn.Linear(MLP_Input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 10)


        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')





        #####quantization code 


        self.quantize=QuantizerFunction(input_dims=256,
                        CodebookSize=128,
                        Method=Method,
                        N_factor=N_factors,
                        alpha=alpha,
                        N_discretizers=3)

        self.to(self.device)
 

    def forward(self,s_t):
        
        '''

        s_t has shape (bsz,n_channels,H,W)
        '''

        x_original=s_t
        bsz,n_channels,H,W=s_t.shape


        s_t=self.CNNencoder(s_t)
        
        bsz,n_channels_,H_,W_=s_t.shape

        s_t=s_t.reshape(bsz,-1,n_channels_*self.patch_size*self.patch_size)
        bsz,T,dim=s_t.shape
        
        ####conduct quantization
        s_t=s_t.reshape(T*bsz,1,dim)

        s_t,ExtraLoss,att_scores=self.quantize(s_t)

        s_t=s_t.reshape(bsz,T,dim).reshape(bsz,n_channels_,H_,W_)
    

        x = s_t.view(bsz,-1)
        x = F.relu(self.fc1(x))
        #x = F.dropout(x, p=self.droprates[1],training=self.training)
        Pred = self.fc2(x)


        ###reconstruction loss

        x_rec=self.CNNdecoder(s_t)



        rec_loss=F.mse_loss(x_original, x_rec)

        return Pred,ExtraLoss,att_scores,rec_loss






#model with is a mxiture of CNN and transformer
class CNNTransformerMixture_quantize(nn.Module):
    """jubrid CNN+Transformer on image model"""
    
    def __init__(self, image_size,output_dim,hidden_dim=2048,
        Method="Original",alpha=0.01,N_factors=1):
        super(CNNTransformerMixture_quantize, self).__init__()

        
        self.image_size=image_size
        self.patch_size=4
        

        #####CNNPart
        self.CNNencoder =CNNEncoder(image_size=image_size)

        self.CNNdecoder=CNNDecoder(Input_N_channels=16)


        if self.image_size==(3,32,32):
            Transformer_intake_height=20
        elif self.image_size==(1,28,28):
            Transformer_intake_height=16
        elif self.image_size==(3,96,96):
            Transformer_intake_height=84

        ####transformer part



        self.transformer=ViT( image_size = Transformer_intake_height,###this actually mean image height
                                channels = 16,
                                patch_size = self.patch_size,
                                num_classes = output_dim,
                                dim = 128,
                                depth = 6,
                                heads = 4,
                                mlp_dim = hidden_dim,
                                dropout = 0.1,
                                emb_dropout = 0.1
                            )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')





        #####quantization code 


        self.quantize=QuantizerFunction(input_dims=256,
                        CodebookSize=128,
                        Method=Method,
                        N_factor=N_factors,
                        alpha=alpha,
                        N_discretizers=3)

        self.to(self.device)
 

    def forward(self,s_t):
        
        '''

        s_t has shape (bsz,n_channels,H,W)
        '''

        x_original=s_t
        bsz,n_channels,H,W=s_t.shape


        s_t=self.CNNencoder(s_t)
        
        bsz,n_channels_,H_,W_=s_t.shape

        s_t=s_t.reshape(bsz,-1,n_channels_*self.patch_size*self.patch_size)
        bsz,T,dim=s_t.shape
        
        ####conduct quantization
        s_t=s_t.reshape(T*bsz,1,dim)

        s_t,ExtraLoss,att_scores=self.quantize(s_t)

        s_t=s_t.reshape(bsz,T,dim).reshape(bsz,n_channels_,H_,W_)
        print("s_t")
        print(s_t.shape)
        Pred=self.transformer(s_t)


        ###reconstruction loss

        x_rec=self.CNNdecoder(s_t)



        rec_loss=F.mse_loss(x_original, x_rec)

        return Pred,ExtraLoss,att_scores,rec_loss



 





if __name__ == "__main__":
    
    img = torch.randn(6, 3, 96, 96)

    # model=Transformer_quantize(image_size=(3,32,32),output_dim=10)
    # preds,Qloss,att_scores = model(img) # (1, 1000)
    # print("preds")
    # print(preds.shape)

    # model=CNNEncoder(image_size=(3,32,32))
    # pred=model(img)
    # print("pred")
    # print(pred.shape)


    # decoder_model=CNNDecoder(16)

    # rec=decoder_model(pred)
    # print("rec")
    # print(rec.shape)


    model=CNNTransformerMixture_quantize(image_size=(3,96,96),output_dim=10,Method="Adaptive_HierachicalMask")
    preds,Qloss,att_scores,rec_loss = model(img) # (1, 1000)
    print("preds")
    print(preds.shape)


    # model=CNNFeature_quantize(image_size=(3,32,32),output_dim=10,Method="Adaptive_HierachicalMask")
    # preds,Qloss,att_scores,rec_loss = model(img) # (1, 1000)
    # print("preds")
    # print(preds.shape)

    # model=CNN_quantize(image_size=(3,96,96),output_dim=10,Method="Adaptive_HierachicalMask")
    # preds,Qloss,att_scores,rec_loss = model(img) # (1, 1000)
    # print("preds")
    # print(preds.shape)