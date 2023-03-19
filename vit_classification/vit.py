import sys
import torch.nn as nn
import torch

sys.path.append("../transformer_captioning") 
from transformer import (
    AttentionLayer,
    MultiHeadAttentionLayer,
    PositionalEncoding,
    SelfAttentionBlock,
    CrossAttentionBlock,
    FeedForwardBlock
)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=2048, dropout=0.1):
        super().__init__()
        self.self_attention = SelfAttentionBlock(d_model, num_heads, dropout=dropout)
        self.feed_forward = FeedForwardBlock(d_model, num_heads, d_ff, dropout=dropout)

    def forward(self, seq, mask):
        x = self.self_attention(seq, mask)
        x = self.feed_forward(x)

        return x

class ViT(nn.Module):
    """
        - A ViT takes an image as input, divides it into patches, and then feeds the patches through a transformer to output a sequence of patch embeddings. 
        - To perform classification with a ViT we patchify the image, embed each patch using an embedding layer and add a learnable [CLS] token to the beginning of the sequence.
        - The output embedding corresponding to the [CLS] token is then fed through a linear layer to obtain the logits for each class.
    """

    def __init__(self, patch_dim, d_model, d_ff, num_heads, num_layers, num_patches, num_classes, device = 'cuda'):
        """
            Construct a new ViT instance.
            Inputs
            - patch_dim: the dimension of each patch
            - d_model: the dimension of the input to the transformer blocks
            - d_ff: the dimension of the intermediate layer in the feed forward block 
            - num_heads: the number of heads in the multi head attention layer
            - num_layers: the number of transformer blocks
            - num_patches: the number of patches in the image
        """

        super().__init__()

        self.patch_dim = patch_dim
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_patches = num_patches
        self.num_classes = num_classes
        self.device = device

        self.patch_embedding = None # TODO (Linear Layer that takes as input a patch and outputs a d_model dimensional vector)
        self.positional_encoding = None # TODO (use the positional encoding from the transformer captioning solution)
        self.fc = None # TODO (takes as input the embedding corresponding to the [CLS] token and outputs the logits for each class)
        self.cls_token = None # TODO (learnable [CLS] token embedding)

        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])

        self.apply(self._init_weights)
        self.device = device 
        self.to(device)

    def patchify(self, images):
        """
            Given a batch of images, divide each image into patches and flatten each patch into a vector.
            Inputs:
                - images: a FloatTensor of shape (N, 3, H, W) giving a minibatch of images
            Returns:
                - patches: a FloatTensor of shape (N, num_patches, patch_dim x patch_dim x 3) giving a minibatch of patches    
        """

        # TODO - Break images into a grid of patches
        # Feel free to use pytorch built-in functions to do this
        
        return images

    def forward(self, images):
        """
            Given a batch of images, compute the logits for each class. 
            Inputs:
                - images: a FloatTensor of shape (N, 3, H, W) giving a minibatch of images
            Returns:
                - logits: a FloatTensor of shape (N, C) giving the logits for each class
        """
        
        patches = self.patchify(images)
        patches_embedded = self.patch_embedding(patches)
        
        output = None # TODO (append a CLS token to the beginning of the sequence of patch embeddings)

        output = self.positional_encoding(patches_embedded)
        mask = torch.ones((self.num_patches, self.num_patches), device=self.device)

        for layer in self.layers:
            output = layer(output, mask)

        output = None # TODO (take the embedding corresponding to the [CLS] token and feed it through a linear layer to obtain the logits for each class)

        return output

    def _init_weights(self, module):
        """
        Initialize the weights of the network.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)




