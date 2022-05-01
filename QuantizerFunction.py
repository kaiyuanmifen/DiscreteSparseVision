import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

from Quantization import Quantize

class QuantizerFunction(nn.Module):
    def __init__(self, input_dims,CodebookSize,Method,N_factor,alpha=0.01,N_discretizers=1):
        super(QuantizerFunction, self).__init__()



        self.CodebookSize=CodebookSize

        ####use the hyperparameter as the maximum codebooksize
        self.all_codebooksizes=[max(CodebookSize//(2**j),4) for j in range(N_discretizers)]#list of codebooksizes

        self.input_dims=input_dims
        
        self.Method=Method
        
        self.hid_dim=32
        self.Quantization_projector=nn.Linear(input_dims, self.hid_dim)#do projection so quantization can be done with mult-factors
        self.Quantization_projector_back=nn.Linear(self.hid_dim,input_dims)

        ####use the hyperparameter as the maximum number of factors 
        self.N_factors=[max(N_factor//(2**j),1) for j in range(N_discretizers)]#list of codebooksizes

        print("*********using quantization setting*******")
        print("method: ",Method)


        import itertools
        self.N_factors_CBsizes=[]

        for r in itertools.product(self.N_factors,self.all_codebooksizes):
            self.N_factors_CBsizes.append([r[0],r[1]])

        self.N_factors_CBsizes.append([self.hid_dim,9999])#the continous version

        if "Mask" in self.Method:
            self.N_factors_CBsizes.append([0,1])#the masked out(all zero) factor


        self.N_tightness_levels=len(self.N_factors_CBsizes)

        self.alpha=alpha###hyperparameter control penalizaing term for using more factors




        if ("Adaptive" in self.Method) and ("Mask" in self.Method) :
          
            
            #self.Quantization_projector=nn.Linear(input_dims, 8)#do projection so quantization can be done with mult-factors
            #self.Quantization_projector_back=nn.Linear(8,input_dims)

            self.QuantizeFunctions=nn.ModuleList([Quantize(self.hid_dim,self.N_factors_CBsizes[o][1],self.N_factors_CBsizes[o][0]) for o in range(self.N_tightness_levels-2)])   
            ###keys for the quantization modules 

            self.quantization_keys=torch.nn.Parameter(torch.randn(self.N_tightness_levels,1,self.hid_dim))

            self.quantization_attention=torch.nn.MultiheadAttention(embed_dim=self.hid_dim, num_heads=4)

        elif "Adaptive" in self.Method:
          
            
            #self.Quantization_projector=nn.Linear(input_dims, 8)#do projection so quantization can be done with mult-factors
            #self.Quantization_projector_back=nn.Linear(8,input_dims)

            self.QuantizeFunctions=nn.ModuleList([Quantize(self.hid_dim,self.N_factors_CBsizes[o][1],self.N_factors_CBsizes[o][0]) for o in range(self.N_tightness_levels-1)])   
            ###keys for the quantization modules 

            self.quantization_keys=torch.nn.Parameter(torch.randn(self.N_tightness_levels,1,self.hid_dim))

            self.quantization_attention=torch.nn.MultiheadAttention(embed_dim=self.hid_dim, num_heads=4)

        elif self.Method=="Quantization":
            self.QuantizeFunctions=Quantize(self.hid_dim,self.CodebookSize,self.N_factors[0])#the total codebook size of the method need to be the same
            

        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        #self.device ='cpu'

        self.to(self.device)



    def forward(self, state):

        ####use lower temperature (sharper softmax), whe evaluating
        if self.training:
            Temperature=1
        else:
            Temperature=0.01


        if self.Method!="Original":
            state=self.Quantization_projector(state)

            bsz,T,Hsz=state.shape



            if self.Method=="Quantization":
                state=state.reshape(bsz*T,1,Hsz)
                state,CBloss,ind=self.QuantizeFunctions(state)#use a fixed function to discretize


                att_scores=torch.zeros(self.N_tightness_levels).unsqueeze(0).unsqueeze(1)

                ExtraLoss=CBloss

            elif self.Method=="Adaptive_Quantization":
                
                ###key-query attention to decide which quantization_function to use
                query=state.reshape(bsz*T,1,Hsz)
                
                _,att_scores=self.quantization_attention(query=query, key=self.quantization_keys,value=self.quantization_keys)

                att_scores=nn.functional.gumbel_softmax(att_scores,hard=True,dim=2,tau=Temperature)
        
                state=state.reshape(bsz*T,1,Hsz)

                Zs=[]
                CBloss=0
                for i in range(self.N_tightness_levels-1):
                    Z,CBloss_vec,_=self.QuantizeFunctions[i](state)#use a fixed function to discretize
                    Zs.append(Z)
                    CBloss+=CBloss_vec#sum up the codebookloss

                Zs.append(state) ####no quantization

                CBloss=CBloss/self.N_tightness_levels
                 


                N_factors_CBsizes_vec=torch.tensor(self.N_factors_CBsizes).to(self.device).float()
                
                N_factors_vec=N_factors_CBsizes_vec[:,0].unsqueeze(0).unsqueeze(0).repeat(1,att_scores.shape[1],1).to(self.device)    
                N_CBsizes_vec=N_factors_CBsizes_vec[:,1].unsqueeze(0).unsqueeze(0).repeat(1,att_scores.shape[1],1).to(self.device)    
                                

                bottlenecking_penanlty=(N_factors_vec*att_scores).mean(dim=1).sum()+(torch.log(N_CBsizes_vec)*att_scores).mean(dim=1).sum()

                ExtraLoss=CBloss+self.alpha*bottlenecking_penanlty


                Zs=torch.cat(Zs,1)
        
                state=torch.bmm(att_scores.permute(1,0,2),Zs)


            elif self.Method=="Adaptive_Hierachical":
                
                ###key-query attention to decide which quantization_function to use
                query=state.reshape(bsz*T,1,Hsz)
                
                _,att_scores=self.quantization_attention(query=query, key=self.quantization_keys,value=self.quantization_keys)

                att_scores=nn.functional.gumbel_softmax(att_scores,hard=True,dim=2,tau=Temperature)

                
                state=state.reshape(bsz*T,1,Hsz)

                Zs=[]
                CBloss=0
                Z=state.clone()
                for i in range(self.N_tightness_levels-1):
                    Z,CBloss_vec,_=self.QuantizeFunctions[i](Z)#use a fixed function to discretize
                    Zs.append(Z)
                    CBloss+=CBloss_vec#sum up the codebookloss

                Zs.append(state) ####no quantization

                CBloss=CBloss/self.N_tightness_levels
                 


                N_factors_CBsizes_vec=torch.tensor(self.N_factors_CBsizes).to(self.device).float()
                
                N_factors_vec=N_factors_CBsizes_vec[:,0].unsqueeze(0).unsqueeze(0).repeat(1,att_scores.shape[1],1).to(self.device)    
                N_CBsizes_vec=N_factors_CBsizes_vec[:,1].unsqueeze(0).unsqueeze(0).repeat(1,att_scores.shape[1],1).to(self.device)    
                                

                bottlenecking_penanlty=(N_factors_vec*att_scores).mean(dim=1).sum()+(torch.log(N_CBsizes_vec)*att_scores).mean(dim=1).sum()
                ExtraLoss=CBloss+self.alpha*bottlenecking_penanlty


                Zs=torch.cat(Zs,1)
        
                state=torch.bmm(att_scores.permute(1,0,2),Zs)

 
            elif self.Method=="Adaptive_HierachicalMask":
                    
                    ###key-query attention to decide which quantization_function to use
                    query=state.reshape(bsz*T,1,Hsz)
                    
                    _,att_scores=self.quantization_attention(query=query, key=self.quantization_keys,value=self.quantization_keys)

                    att_scores=nn.functional.gumbel_softmax(att_scores,hard=True,dim=2,tau=Temperature)

                    
                    state=state.reshape(bsz*T,1,Hsz)

                    Zs=[]
                    CBloss=0
                    Z=state.clone()
                    for i in range(self.N_tightness_levels-2):
                        Z,CBloss_vec,_=self.QuantizeFunctions[i](Z)#use a fixed function to discretize
                        Zs.append(Z)
                        CBloss+=CBloss_vec#sum up the codebookloss

                    Zs.append(state) ####no quantization
                    Zs.append(torch.zeros(state.shape).to(self.device))### mask the information all as zeros

                    CBloss=CBloss/self.N_tightness_levels
                     


                    N_factors_CBsizes_vec=torch.tensor(self.N_factors_CBsizes).to(self.device).float()
                    
                    N_factors_vec=N_factors_CBsizes_vec[:,0].unsqueeze(0).unsqueeze(0).repeat(1,att_scores.shape[1],1).to(self.device)    
                    N_CBsizes_vec=N_factors_CBsizes_vec[:,1].unsqueeze(0).unsqueeze(0).repeat(1,att_scores.shape[1],1).to(self.device)    
                                    

                    bottlenecking_penanlty=(N_factors_vec*att_scores).mean(dim=1).sum()+(torch.log(N_CBsizes_vec)*att_scores).mean(dim=1).sum()
                    ExtraLoss=CBloss+self.alpha*bottlenecking_penanlty


                    Zs=torch.cat(Zs,1)
            
                    state=torch.bmm(att_scores.permute(1,0,2),Zs)

            state=state.reshape(bsz,T,Hsz)

            state=self.Quantization_projector_back(state)###shape it back



        elif self.Method=="Original":
 
            CBloss=torch.zeros(1).to(self.device)
            ExtraLoss=CBloss

            att_scores=torch.zeros(self.N_tightness_levels).unsqueeze(0).unsqueeze(1)

        # print('CBloss')
        # print(CBloss)
        # print("bottlenecking_penanlty")
        # print(bottlenecking_penanlty)
        # print("self.N_tightness_levels")
        # print(self.N_tightness_levels)
        # print("att_scores")
        # print(att_scores[0,0,:])
        return state,ExtraLoss,att_scores

