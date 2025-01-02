import torch
from torch import nn
import timm
from transformers import DistilBertModel, DistilBertConfig
import config as CFG

class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(self, model_name=CFG.model_name, pretrained=CFG.pretrained, trainable=CFG.trainable):
        super().__init__() # `super()` is used to call the constructor of the parent class (nn.Module).This ensures this class is properly initialized as a nn.Module. 
        self.model = timm.create_model(model_name, pretrained, num_classes=0, global_pool='avg')
        for p in self.model.parameters():
            p.requires_grad = trainable
    
    def forward(self, x):
        return self.model(x)

class TextEncoder(nn.Module):
    def __init__(self, model_name=CFG.text_encoder_model, pretrained=CFG.pretrained, trainable=CFG.trainable):
        super().__init__()
        if pretrained:
            self.model = DistilBertModel.from_pretrained(model_name)
        else:
            self.model = DistilBertModel(config=DistilBertConfig())
        
        for p in self.model.parameters():
            p.requires_grad = trainable
        
        # we are using the CLS token hidden representation as sentence's embedding
        # Yu: this sets the index of the target token to 0. In Bert-based models, the CLS token (classification token) is typically the first token in the sequence and its index is 0.
        # The CLS token is used as a representation of the entire input sequence.
        self.target_token_idx = 0 # Yu:?

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state

        # last_hidden_state is a tensor with shape (batch_size, seq_len, hidden_size),
        # : (first position) selects all batches,
        # self.target_token_idx (second position) selects the CLS token (the first token in the sequence),
        # : (third position) selects all hidden dimensions
        # therefore, thus function returns the hidden state of the CLS token for each input in the batch

        return last_hidden_state[:, self.target_token_idx, :]

class ProjectionHead(nn.Module):
    def __init__(
            self,
            embedding_dim, 
            projection_dim=CFG.projection_dim,
            dropout=CFG.dropout
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim) # a linear layer that projects the input embeddings from the embedding_dim to the projection_dim
        self.gelu = nn.GELU() # A GELU(Gaussian Error Linear Unit) activation function.
        self.fc = nn.Linear(projection_dim, projection_dim) # another linear layer that maintains the 'project_dim'
        self.dropout = nn.Dropout(dropout) # a dropout layer for regularization
        self.layer_norm = nn.LayerNorm(projection_dim) # a layer normalization layer to normalize the output

    def forward(self, x):

        # x is the input tensor
        projected = self.projection(x)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected # adds the original projected embeddings to the output(residual connection)
        x = self.layer_norm(x) # applies layer normalization to the output

        return x
    