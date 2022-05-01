#!/bin/bash





# #those without contrastive elements

declare -a all_data=("CIFAR10")


#declare -a all_methods=("Transformer_HAQuantizationMask")
declare -a all_methods=("Transformer_HAQuantizationMask" "Transformer_NoQuantization" "Transformer_Quantization" "Transformer_HAQuantization" "CNN_NoQuantization" "CNN_Quantization" "CNN_HAQuantization" "CNN_HAQuantizationMask")

declare -a all_rounds=(1)


declare -a all_contrastive=("no")


for data in "${all_data[@]}"
do

	for method in "${all_methods[@]}"
	do

		for contrastive in "${all_contrastive[@]}"
		do

			for round in "${all_rounds[@]}"
			do

				sbatch JobSubmit.sh $data $method $contrastive $round	
			done
	
		done
	done
done




# #####those withcontrastive elements

# declare -a all_data=("MNIST CIFAR10" "SVHN")
# #declare -a all_data=("CIFAR10")

# declare -a all_methods=("Hybrid_HAQuantizationMask" "Hybrid_NoQuantization" "Hybrid_Quantization" "Hybrid_HAQuantization" "CNNFeatures_NoQuantization" "CNNFeatures_Quantization" "CNNFeatures_HAQuantization" "CNNFeatures_HAQuantizationMask")

# declare -a all_rounds=(1)


# declare -a all_contrastive=("yes" "no")


# for data in "${all_data[@]}"
# do

# 	for method in "${all_methods[@]}"
# 	do

# 		for contrastive in "${all_contrastive[@]}"
# 		do

# 			for round in "${all_rounds[@]}"
# 			do

# 				sbatch JobSubmit.sh $data $method $contrastive $round	
# 			done
	
# 		done
# 	done
# done




####analyze model behaviours



# ############GFN based models
# declare -a all_data=("CIFAR10")
# #declare -a all_methods=("Transformer_HAQuantizationMask" "Transformer_NoQuantization" "Transformer_Quantization" "Transformer_HAQuantization" "CNN_NoQuantization" "CNN_Quantization" "CNN_HAQuantization" "CNN_HAQuantizationMask" "Hybrid_HAQuantizationMask" "Hybrid_NoQuantization" "Hybrid_Quantization" "Hybrid_HAQuantization" "CNNFeatures_NoQuantization" "CNNFeatures_Quantization" "CNNFeatures_HAQuantization" "CNNFeatures_HAQuantizationMask")
# #declare -a all_methods=("Transformer_HAQuantizationMask" "Transformer_NoQuantization" "Transformer_Quantization" "Transformer_HAQuantization" "CNN_NoQuantization" "CNN_Quantization" "CNN_HAQuantization" "CNN_HAQuantizationMask" "Hybrid_HAQuantizationMask" "Hybrid_NoQuantization" "Hybrid_Quantization" "Hybrid_HAQuantization" "CNNFeatures_NoQuantization" "CNNFeatures_Quantization" "CNNFeatures_HAQuantization" "CNNFeatures_HAQuantizationMask")
# declare -a all_methods=("Transformer_HAQuantizationMask" "Transformer_NoQuantization" "Transformer_Quantization" "Transformer_HAQuantization" "CNN_NoQuantization" "CNN_Quantization" "CNN_HAQuantization" "CNN_HAQuantizationMask")


# #declare -a all_methods=("Transformer_HAQuantizationMask")

# declare -a all_rounds=(1)
# #declare -a all_contrastive=("yes" "no")
# declare -a all_contrastive=("no")



# declare -a all_rounds=(1)


# for data in "${all_data[@]}"
# do

# 	for method in "${all_methods[@]}"
# 	do

# 		for contrastive in "${all_contrastive[@]}"
# 		do


# 			for round in "${all_rounds[@]}"
# 			do

# 				./JobSubmit_behavior.sh $data $method contrastive $round	
# 			done
		
# 		done
# 	done
# done