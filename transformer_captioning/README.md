# Image Captioning with Transformers

Please attempt this question sequentially, as the parts build upon previous sections. We will build a transformer decoder and use it for image captioning. Please read 
["Attention Is All You Need"](https://arxiv.org/abs/1706.03762), which introduced transformers.  

## Q.1.a : Attention 

Complete the TODOs in the `AttentionLayer` class in the file `transformer.py`





Given query $q$, key $k$ and value $v$, these are first projected into the same embedding dimenion using separate linear projections. 

The attention output is then given by : 

$$Y = \text{dropout}\bigg(\text{softmax}\bigg(\frac{Q.K^\top + M}{\sqrt{d}}\bigg)\bigg)V$$

where Q, K and V are matrices containing rows of projected queries, keys and values respectively. M is an additive mask, which is used to restrict where attention is applied.

## Q.1.b : Multi-head Attention 

Complete the TODOs in the `MultiHeadAttentionLayer` class in the file `transformer.py`

For the model to have more expressivity, we can add more heads to allow it to attend to different parts of the input. 
For this we split the query, key and value matrices Q,K,V along the embedding dimension, and attention is performed on each of these separately. 
For the ith head, the output is given by : 

$$Y_i = \text{dropout}\bigg(\text{softmax}\bigg(\frac{Q_i.K_i^\top + M}{\sqrt{d/h}}\bigg)\bigg)V_i$$

where $Y_i\in\mathbb{R}^{\ell \times d/h}$, where $\ell$ is our sequence length.

These are then concatenated and projected to the embedding dimension to obtain the overall output:
$$Y = [Y_1;\dots;Y_h]A$$


## Q.1.c : Positional Encoding 


Complete the TODOs in the `PositionalEncoding` class in the file `transformer.py`

While transformers can aggregate information from across the sequence, we need to provide information about the ordering of the tokens. This can be done using a special code for each token, which is precomputed and fixed, and added to the sequence. 


## Q.1.d : Transformer Decoder Layer


Complete the TODOs in the `SelfAttentionBlock`,  `CrossAttentionBlock` and `FeedForwardBlock` classes in the file `transformer.py`

A transformer decoder layer consists of three blocks - a masked self-attention block, a cross-attention block that uses conditioning features (no mask), and a feedforward block, as described in the paper  ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762). The structure of these blocks is explained in the code comments, and also in the paper. The transformer decoder is formed by stacking a number of such layers together. 


## Q.1.e : Transformer Decoder \& Captioning

Most of the implementation for this class has been provided, including auto-regressively predicting caption tokens. Please fill out remaining TODOs in `TransformerDecoder` in the file transformer.py, relating to projecting the captions and features into `embed_dim` dimensions, and getting the causal mask. Also fill out the TODOs in `Trainer` in the file trainer.py for computing the loss between predictions and labels. 

Now, run run.py with the following configurations -
1) num_heads : 2, num_layers : 2, learning_rate : 1e-4
2) num_heads : 4, num_layers : 6, learning_rate : 1e-4
3) num_heads : 4, num_layers : 6, learning_rate : 1e-3

Include loss plot and 2 images from the training set for each at the end of 100 epochs. These models don't perform well on the validation set since we're training on a small subset of the training data.  