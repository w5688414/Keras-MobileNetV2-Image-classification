## train
默认训练group0,可通过 --train --valid来改变训练的数据集和验证的数据集
 python train.py --classes 34 


 ## fine tuning

python train.py --classes 34 --weights ./model/weights_group0_train.h5 --tclasses 34 --train group1_train.txt --valid group1_valid.txt


 python train.py --classes 34 --weights ./model/weights_group1_train.h5 --tclasses 34 --train group2_train.txt --valid group2_valid.txt

 python train.py --classes 34 --weights ./model/weights_group2_train.h5 --tclasses 34 --train group3_train.txt --valid group3_valid.txt

 python train.py --classes 34 --weights ./model/weights_group3_train.h5 --tclasses 34 --train group4_train.txt --valid group4_valid.txt

python train.py --classes 34 --weights ./model/weights_group4_train.h5 --tclasses 34 --train group5_train.txt --valid group5_valid.txt

python train.py --classes 34 --weights ./model/weights_group5_train.h5 --tclasses 34 --train group6_train.txt --valid group6_valid.txt

 python train.py --classes 34 --weights ./model/weights_group6_train.h5 --tclasses 34 --train group7_train.txt --valid group7_valid.txt

 python train.py --classes 34 --weights ./model/weights_group7_train.h5 --tclasses 34 --train group8_train.txt --valid group8_valid.txt

python train.py --classes 34 --weights ./model/weights_group8_train.h5 --tclasses 34 --train group9_train.txt --valid group9_valid.txt
