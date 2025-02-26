


## Training

To pretrain DiffIR_S1, run
```
sh trainS1.sh
```

To train DiffIR_S2, run
```
#set the 'pretrain_network_g' and 'pretrain_network_S1' in ./options/train_DiffIRS2_sino_nomin.yml to be the path of DiffIR_S1's pre-trained model

sh trainS2.sh
```


## Evaluation



- Testing
```
# modify the dataset path in ./options/test_DiffIRS2.yml

sh test.sh 
```



