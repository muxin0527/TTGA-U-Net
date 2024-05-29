#!/bin/bash


while getopts 'c:n:t:r:p' OPT; do
    case $OPT in
        c) cuda=$OPTARG;;
        n) name=$OPTARG;;
	t) task=$OPTARG;;
        r) train="false";;
        p) predict="true";;
        
    esac
done
echo $name	


if ${predict}
then

	CUDA_VISIBLE_DEVICES=${cuda} unetr_pp_predict -i 0.7 -o gaussion/predict_0.7 -m 3d_fullres -t ${task} -f 0 -chk model_best -tr unetr_pp_trainer_${name}
fi



