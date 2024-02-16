# DPAR: Decoupled Graph Neural Networks with Node-Level Differential Privacy

 Accepted at 2024 ACM International World Wide Web Conference (WWW 24)

## Install metis on mac:  

1. Download metis-5.1.0.tar.gz from http://glaros.dtc.umn.edu/gkhome/metis/metis/download and unpack it
2. cd metis-5.1.0
3. make config shared=1
4. make install
5. export METIS_DLL=/usr/local/lib/libmetis.dylib


## Package version:
tensorflow==1.15.4  
networkx==2.5  
metis==0.2a4  
pynverse==0.1.4.4  
scikit-learn==0.23.2  
scipy==1.5.2  
numba==0.51.2  
numpy==1.18.5  


## Run main.py  
bash script example
```bash
#!/bin/bash

alpha_list=(0.99)
topk_list=(4 8 16 32 64 128 256 512 1024)

for alpha_value in "${alpha_list[@]}"
do
	for topk_value in "${topk_list[@]}"
	do
 		python run_demo_test_noPowerInteration.py --lr 5e-4 --topk $topk_value --ntrain_div_classes 3754 --alpha $alpha_value
	done
done
```

## Dataset options

1. Cora: ```data/cora_ml```
2. Pubmed: ```data/pubmed```
3. Ms academic: ```data/ms_academic```
4. Facebook: ```data/facebook```
5. pubmed: ```data/pubmed```
6. Physics: ```data/physics```
