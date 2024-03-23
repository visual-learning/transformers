# Classification with Vision Transformers

The previous question used attention over image features provided in the COCO dataset. But what if we want to use attention directly over the image? Recent works have explored this, and in this question we will implement a Vision Transformer (ViT), as described in this [paper](https://arxiv.org/pdf/2010.11929.pdf), for the task of image classification. 

## Q.2.a : Initialization 

Complete the TODOs in the `__init__` function in the file `vit.py`. In addition to projection and encoding layers, this includes a class token. This is a learnable parameter, which you will use in part c. 


## Q.2.b : Patchification

Complete the TODOs in the `patchify` function in the file `vit.py`

The vision transformer breaks the image into a set of patches, each of which is then projected into a corresponding token to be attended over. This allows the transformer to learn representations using attention directly from pixels. 


## Q.2.c : ViT Forward Pass


Complete the TODOs in the `forward` function in the file `vit.py`

This includes utlizing the class token. This should be included at the beginning of the sequence before being passed to the transformer. The class prediction only uses the first token from the output sequence, which corresponds to the class token. 


## Q.2.d : Loss \& Classification

Complete the TODO in the `loss` function in the file `trainer.py` 

Create a directory for saving model checkpoints using `mkdir checkpoints`.

After this, train the model on CIFAR10 using run.py. Include the train and test accurary, and the training loss in your hw pdf submission. Note that on datasets of this small size, training a ViT from scratch as we're done here does not yield better results than using a convolutional network. 



