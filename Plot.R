

AllFiles=list.files('Results/')

AllFiles=AllFiles[grepl(AllFiles,pattern = "csv")]


AllFiles=AllFiles[grepl(AllFiles,pattern = "True|False")]


Data=NULL

for (File in AllFiles){

  Vec=read.csv(paste0("Results/",File))
  names(Vec)[1]="Epoch"

  Infor=strsplit(File,split = "_")[[1]]
  
  Vec$Seed=as.integer(Infor[length(Infor)-1])
  
  
  #if (!is.na(as.integer(Infor[length(Infor)-3]))){
  
  
  
  Vec$Data=Infor[3]
  
  Vec$Method=Infor[2]
  
  Vec$Model=Infor[1]
  
  Vec$constrastive=Infor[4]
  
  Vec$Exp=paste0(Vec$Method,"_",Vec$constrastive)
  

  
  Data=rbind(Data,Vec)
  
  
  #}
  
}



head(Data)

unique(Data$Model)
Data$GeneralizationGap=Data$test_acc-Data$test_acc_OOD

#####task

library(ggplot2)
names(Data)
head(Data)
#Data=Data[Data$Method!="MLP_SVD",]

for (DataNames in c("MNIST","SVHN", "CIFAR10")){
  
  for (Model in c("CNN","Transformer","CNNFeatures","Hybrid")){

Exp=paste0(DataNames,"_",Model)


VecPlot=Data
VecPlot=VecPlot[(VecPlot$Data==DataNames)&(Data$Model==Model),]

Plot <- ggplot(VecPlot, aes(x=Epoch, y=test_acc,color=Exp)) +geom_smooth()+
  ggtitle(DataNames)


ggsave(plot = Plot,paste0('images/',Exp,"_testacc.png"),scale=3)


Plot <- ggplot(VecPlot, aes(x=Epoch, y= log(Total_train_loss) ,color=Exp)) +geom_smooth()+
  ggtitle(DataNames)


ggsave(plot = Plot,paste0('images/',Exp,"_ trainloss .png"),scale=3)


head(VecPlot)
Plot <- ggplot(VecPlot, aes(x=Epoch, y=test_acc_OOD,color=Exp)) +geom_smooth()+
  ggtitle(DataNames)


ggsave(plot = Plot,paste0('images/',Exp,"_testaccOOD.png"),scale=3)


Plot <- ggplot(VecPlot, aes(x=Epoch, y= Quantization_losses ,color=Exp)) +geom_smooth()+
  ggtitle(DataNames)



ggsave(plot = Plot,paste0('images/',Exp,"_Quantization_losses.png"),scale=3)




Plot <- ggplot(VecPlot, aes(x=Epoch, y= constrastive_loss ,color=Exp)) +geom_smooth()+
  ggtitle(DataNames)



ggsave(plot = Plot,paste0('images/',Exp,"_constrastive_loss.png"),scale=3)





Plot <- ggplot(VecPlot, aes(x=Epoch, y=task_losses ,color=Exp)) +geom_smooth()+
  ggtitle(DataNames)



ggsave(plot = Plot,paste0('images/',Exp,"_task_lossess.png"),scale=3)


Plot <- ggplot(VecPlot, aes(x=Epoch, y=GeneralizationGap ,color=Exp)) +geom_smooth()+
  ggtitle(DataNames)



ggsave(plot = Plot,paste0('images/',Exp,"_GeneralizationGap.png"),scale=3)

}

}









####model behaviour analysis



AllFiles=list.files('Results/BehaviourAnalysis/')

AllFiles=AllFiles[grepl(AllFiles,pattern = "csv")]

AllFiles=AllFiles[grepl(AllFiles,pattern = "Colorjitter|Solarize|Gaussian" )]

Data=NULL

for (File in AllFiles){
  
  Vec=read.csv(paste0("Results/BehaviourAnalysis/",File))
  names(Vec)[1]="noise"
  
  Infor=strsplit(File,split = "_")[[1]]
  
  Vec$Seed=as.integer(Infor[5])
  
  
  #if (!is.na(as.integer(Infor[length(Infor)-3]))){
  

  
  Vec$Data=Infor[3]
  
  Vec$Method=Infor[2]
  
  Vec$Model=Infor[1]
  Vec$contrastive=Infor[4]
  Vec$Exp=paste0(Vec$Method,"_",Vec$contrastive)
  
  Vec$Augmentation=Infor[6]
  
  
  Data=rbind(Data,Vec)
  
  
  #}
  
}

tail(Data)

#Data$Quantization_losses=as.numeric(unlist(lapply(strsplit(Data$Quantization_losses,split = "[(]|[)]"),function(x){x[[2]]})))

unique(Data$Augmentation)
library(ggplot2)

#Data=Data[Data$Method!="MLP_SVD",]

for (DataNames in c("CIFAR10")){
  
  for (Model in c("Transformer","CNN")){
    
    for (Augmentation in c("Gaussian","Solarize",'Colorjitter')){
    
    Exp=paste0(DataNames,"_",Model,"_",Augmentation)
    
    
    VecPlot=Data
    VecPlot=VecPlot[(VecPlot$Data==DataNames)&(Data$Model==Model)&(Data$Augmentation==Augmentation),]
    
    
    if (Augmentation=="Random"){
      
      Plot <- ggplot(VecPlot, aes(x=Exp, y=test_acc)) +geom_boxplot()+
        ggtitle(DataNames)
      
      
      ggsave(plot = Plot,paste0('images/Behaviors/',Exp,"_Behaviour.png"),scale=3)
      
    }
    else {
    Plot <- ggplot(VecPlot, aes(x=noise, y=test_acc,color=Exp)) +geom_smooth()+
      ggtitle(DataNames)
    
    
    ggsave(plot = Plot,paste0('images/Behaviors/',Exp,"_Behaviour.png"),scale=3)
    
    
    Plot <- ggplot(VecPlot, aes(x=noise, y=Quantization_losses,color=Exp)) +geom_smooth()+
      ggtitle(DataNames)
    
    
    ggsave(plot = Plot,paste0('images/Behaviors/',Exp,"_BehaviourQloss.png"),scale=3)
    }
    
    }
  
  }
  
}


####usage of different discretizer

for (DataNames in c("CIFAR10")){
  
  for (Model in c("Transformer","CNN","CNNFeatures","Hybrid")){
    
    
    for (Method in c("HAQuantization","HAQuantizationMask")){
      
     
    Exp=paste0(DataNames,"_",Model,"_",Method)
    
    
    VecPlot=Data
    VecPlot=VecPlot[(VecPlot$Data==DataNames)&(Data$Model==Model)&(Data$Method==Method),]
    
    VQattScores=NULL
    TightnessScore=c()
    
    
    for (i in 1:nrow(VecPlot)){
      
      Vec=strsplit(VecPlot$Qatt_scores[i],split = "[[]|[,]|[]]")[[1]]
      Vec=as.numeric(Vec[2:length(Vec)])
      if (Method=="HAQuantization"){
        Vec=c(Vec[length(Vec)],Vec[-length(Vec)])
      }
      
      if (Method=="HAQuantizationMask"){
        Vec=c(Vec[length(Vec)-1],Vec[-(length(Vec)-1)])
      }
      
      
      VQattScores=rbind(VQattScores,data.frame(Noise=VecPlot$noise[i],Tightness=1:length(Vec),NormCount=Vec))
      
      
      TightnessScore=c(TightnessScore,sum(as.numeric(as.character(VQattScores$Tightness))*VQattScores$NormCount))
    }
    
    VQattScores$Tightness=as.factor(VQattScores$Tightness)
    VQattScores$Noise=as.factor(VQattScores$Noise)
    VQattScores$NormCount=as.numeric(VQattScores$NormCount)
    
    Plot <- ggplot(VQattScores, aes(x=Noise, y=NormCount,fill=Tightness)) +
      geom_bar(position="dodge", stat="identity")+
      ggtitle(DataNames)
    
    ggsave(plot = Plot,paste0('images/Behaviors/',Exp,"_BehaviourUssage.png"),scale=3)
    
    VecPlot$BottleneckTightness=TightnessScore
    
    Plot <- ggplot(VecPlot, aes(x=noise, y=BottleneckTightness)) +geom_point()+
      ggtitle(DataNames)
    
    ggsave(plot = Plot,paste0('images/Behaviors/',Exp,"_noise_tightness.png"),scale=3)
    
    
    
  }
  
  }
}

