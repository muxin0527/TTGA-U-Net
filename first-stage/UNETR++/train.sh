#!/bin/bash


while getopts 'c:n:t:r:p' OPT; do
    case $OPT in
        c) cuda=$OPTARG;;
        n) name=$OPTARG;;
	t) task=$OPTARG;;
        r) train="true";;
        p) predict="false";;
        
    esac
done
echo $name	


if ${train}
then
	
	cd /home/data/Program/unetr_plus_plus/unetr_pp/
	CUDA_VISIBLE_DEVICES=${cuda} unetr_pp_train 3d_fullres unetr_pp_trainer_${name} ${task} 0
fi


