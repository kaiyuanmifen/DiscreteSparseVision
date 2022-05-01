######these plots are for the paper



################Figure for generalization gap vs. epochs




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

for (DataNames in c("CIFAR10")){
  
  for (Model in c("CNN","Transformer")){
    
    Exp=paste0(DataNames,"_",Model)
    
    
    VecPlot=Data
    VecPlot=VecPlot[(VecPlot$Data==DataNames)&(Data$Model==Model)&(Data$Method%in%c("Quantization","NoQuantization")),]
    
   
    Plot <- ggplot(VecPlot, aes(x=Epoch, y=GeneralizationGap ,color=Method)) +geom_smooth()+
      ggtitle(DataNames)
    
    
    
    ggsave(plot = Plot,paste0('../ImagesForPaper/',Exp,"_GeneralizationGap.png"),scale=1.5)
    
  }
  
}

















######Figure for Robustnes on different level of noise


AllFiles=list.files('Results/BehaviourAnalysis/')

AllFiles=AllFiles[grepl(AllFiles,pattern = "csv")]


Data=NULL

for (File in AllFiles){
  
  Vec=read.csv(paste0("Results/BehaviourAnalysis/",File))
  names(Vec)[1]="noise"
  
  Infor=strsplit(File,split = "_")[[1]]
  
  Vec$Seed=as.integer(Infor[length(Infor)-1])
  
  
  #if (!is.na(as.integer(Infor[length(Infor)-3]))){
  
  Vec$Generalization_Gap=Vec$test_acc-Vec$test_acc[1]
  
  Vec$Data=Infor[3]
  
  Vec$Method=Infor[2]
  
  Vec$Model=Infor[1]
  Vec$contrastive=Infor[4]
  Vec$Algorithm=paste0(Vec$Method,"_",Vec$contrastive)
  
  
  
  Data=rbind(Data,Vec)
  
  
  #}
  
}

tail(Data)
unique(Data$Model)
Data=Data[(Data$Model%in%c("CNN","Transformer"))&(Data$Method%in%c("Quantization","NoQuantization")),]

#Data$Quantization_losses=as.numeric(unlist(lapply(strsplit(Data$Quantization_losses,split = "[(]|[)]"),function(x){x[[2]]})))

unique(Data$Method)
library(ggplot2)

#Data=Data[Data$Method!="MLP_SVD",]

for (DataNames in c("CIFAR10")){
  
  for (Model in c("CNN","Transformer")){
    
    Exp=paste0(DataNames,"_",Model)
    
    
    VecPlot=Data
    VecPlot=VecPlot[(VecPlot$Data==DataNames)&(Data$Model==Model),]
    
    Plot <- ggplot(VecPlot, aes(x=noise, y=Generalization_Gap,color=Method)) +geom_line()+
      ggtitle(Exp)
    
    
    ggsave(plot = Plot,paste0('../ImagesForPaper/',Exp,"_Robustness.png"),scale=1)
    
    
    
  }
  
}






####figure show that hierachial adaptive version further improves generalization gap



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

unique(Data$Method)
Data$GeneralizationGap=Data$test_acc-Data$test_acc_OOD


library(ggplot2)
names(Data)
head(Data)
#Data=Data[Data$Method!="MLP_SVD",]

for (DataNames in c("CIFAR10")){
  
  for (Model in c("CNN","Transformer")){
    
    Exp=paste0(DataNames,"_",Model)
    
    
    VecPlot=Data
    VecPlot=VecPlot[(VecPlot$Data==DataNames)&(Data$Model==Model)&(Data$Method%in%c("Quantization","NoQuantization","HAQuantization")),]
    
    
    Plot <- ggplot(VecPlot, aes(x=Epoch, y=GeneralizationGap ,color=Method)) +geom_smooth()+
      ggtitle(DataNames)
    
    
    
    ggsave(plot = Plot,paste0('../ImagesForPaper/',Exp,"_GeneralizationGapHA.png"),scale=1.5)
    
  }
  
}









######Figure for Robustnes on different level of noise, including HA 


AllFiles=list.files('Results/BehaviourAnalysis/')

AllFiles=AllFiles[grepl(AllFiles,pattern = "csv")]


Data=NULL

for (File in AllFiles){
  
  Vec=read.csv(paste0("Results/BehaviourAnalysis/",File))
  names(Vec)[1]="noise"
  
  Infor=strsplit(File,split = "_")[[1]]
  
  Vec$Seed=as.integer(Infor[length(Infor)-1])
  
  
  #if (!is.na(as.integer(Infor[length(Infor)-3]))){
  
  Vec$Generalization_Gap=Vec$test_acc-Vec$test_acc[1]
  
  Vec$Data=Infor[3]
  
  Vec$Method=Infor[2]
  
  Vec$Model=Infor[1]
  Vec$contrastive=Infor[4]
  Vec$Algorithm=paste0(Vec$Method,"_",Vec$contrastive)
  
  
  
  Data=rbind(Data,Vec)
  
  
  #}
  
}

tail(Data)
unique(Data$Model)
Data=Data[(Data$Model%in%c("CNN","Transformer"))&(Data$Method%in%c("Quantization","NoQuantization","HAQuantization")),]

#Data$Quantization_losses=as.numeric(unlist(lapply(strsplit(Data$Quantization_losses,split = "[(]|[)]"),function(x){x[[2]]})))

unique(Data$Method)
library(ggplot2)

#Data=Data[Data$Method!="MLP_SVD",]

for (DataNames in c("CIFAR10")){
  
  for (Model in c("CNN","Transformer")){
    
    Exp=paste0(DataNames,"_",Model)
    
    
    VecPlot=Data
    VecPlot=VecPlot[(VecPlot$Data==DataNames)&(Data$Model==Model),]
    
    Plot <- ggplot(VecPlot, aes(x=noise, y=Generalization_Gap,color=Method)) +geom_line()+
      ggtitle(Exp)
    
    
    ggsave(plot = Plot,paste0('../ImagesForPaper/',Exp,"_RobustnessHA.png"),scale=1)
    
    
    
  }
  
}





####figure show that sparsity (mask) give better performance in transformer



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

unique(Data$Method)
Data$GeneralizationGap=Data$test_acc-Data$test_acc_OOD


library(ggplot2)
names(Data)
head(Data)
#Data=Data[Data$Method!="MLP_SVD",]

for (DataNames in c("CIFAR10")){
  
  for (Model in c("CNN","Transformer")){
    
    Exp=paste0(DataNames,"_",Model)
    
    
    VecPlot=Data
    VecPlot=VecPlot[(VecPlot$Data==DataNames)&(Data$Model==Model)&(Data$Method%in%c("Quantization","NoQuantization","HAQuantization","HAQuantizationMask")),]
    
    
    Plot <- ggplot(VecPlot, aes(x=Epoch, y=GeneralizationGap ,color=Method)) +geom_smooth()+
      ggtitle(DataNames)
    
    
    
    ggsave(plot = Plot,paste0('../ImagesForPaper/',Exp,"_GeneralizationGapHAMASK.png"),scale=1.5)
    
  }
  
}






######Figure for Robustnes on different level of noise, including HAMask 


AllFiles=list.files('Results/BehaviourAnalysis/')

AllFiles=AllFiles[grepl(AllFiles,pattern = "csv")]


Data=NULL

for (File in AllFiles){
  
  Vec=read.csv(paste0("Results/BehaviourAnalysis/",File))
  names(Vec)[1]="noise"
  
  Infor=strsplit(File,split = "_")[[1]]
  
  Vec$Seed=as.integer(Infor[length(Infor)-1])
  
  
  #if (!is.na(as.integer(Infor[length(Infor)-3]))){
  
  Vec$Generalization_Gap=Vec$test_acc-Vec$test_acc[1]
  
  Vec$Data=Infor[3]
  
  Vec$Method=Infor[2]
  
  Vec$Model=Infor[1]
  Vec$contrastive=Infor[4]
  Vec$Algorithm=paste0(Vec$Method,"_",Vec$contrastive)
  
  
  
  Data=rbind(Data,Vec)
  
  
  #}
  
}

tail(Data)
unique(Data$Method)
Data=Data[(Data$Model%in%c("CNN","Transformer"))&(Data$Method%in%c("Quantization","NoQuantization","HAQuantization","HAQuantizationMask")),]

#Data$Quantization_losses=as.numeric(unlist(lapply(strsplit(Data$Quantization_losses,split = "[(]|[)]"),function(x){x[[2]]})))

unique(Data$Method)
library(ggplot2)

#Data=Data[Data$Method!="MLP_SVD",]

for (DataNames in c("CIFAR10","MNIST")){
  
  for (Model in c("CNN","Transformer")){
    
    Exp=paste0(DataNames,"_",Model)
    
    
    VecPlot=Data
    VecPlot=VecPlot[(VecPlot$Data==DataNames)&(Data$Model==Model),]
    
    Plot <- ggplot(VecPlot, aes(x=noise, y=Generalization_Gap,color=Method)) +geom_line()+
      ggtitle(Exp)
    
    
    ggsave(plot = Plot,paste0('../ImagesForPaper/',Exp,"_RobustnessHAMask.png"),scale=1)
    
    
    
  }
  
}




####usage of different discretizer




AllFiles=list.files('Results/BehaviourAnalysis/')

AllFiles=AllFiles[grepl(AllFiles,pattern = "csv")]


Data=NULL

for (File in AllFiles){
  
  Vec=read.csv(paste0("Results/BehaviourAnalysis/",File))
  names(Vec)[1]="noise"
  
  Infor=strsplit(File,split = "_")[[1]]
  
  Vec$Seed=as.integer(Infor[length(Infor)-1])
  
  
  #if (!is.na(as.integer(Infor[length(Infor)-3]))){
  
  Vec$Generalization_Gap=Vec$test_acc-Vec$test_acc[1]
  
  Vec$Data=Infor[3]
  
  Vec$Method=Infor[2]
  
  Vec$Model=Infor[1]
  Vec$contrastive=Infor[4]
  Vec$Algorithm=paste0(Vec$Method,"_",Vec$contrastive)
  
  
  
  Data=rbind(Data,Vec)
  
  
  #}
  
}


for (DataNames in c("CIFAR10")){
  
  for (Model in c("Transformer","CNN")){
    
    
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
      
      ggsave(plot = Plot,paste0('../ImagesForPaper/',Exp,"_BehaviourUssage.png"),scale=2)
      
      VecPlot$BottleneckTightness=TightnessScore
      
      Plot <- ggplot(VecPlot, aes(x=noise, y=BottleneckTightness)) +geom_point()+
        ggtitle(DataNames)
      
      ggsave(plot = Plot,paste0('../ImagesForPaper/',Exp,"_noise_tightness.png"),scale=1.5)
      
      
      
    }
    
  }
}




