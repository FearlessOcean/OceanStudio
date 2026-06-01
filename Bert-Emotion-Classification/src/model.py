from torch import nn
import torch
import config

class ReviewAnalyzeModel(nn.Module):
    def __init__(self,cocab_size,padding_index):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings = cocab_size ,
            embedding_dim = config.EMBEDDING_DIM,
            padding_idx= padding_index )
        self.gru = nn.GRU(
            input_size=config.EMBEDDING_DIM,
            hidden_size=config.HIDDEN_SIZE,
            batch_first=True
        )
        self.linear = nn.Linear(config.HIDDEN_SIZE,1)

    def forward(self,x:torch.tensor):
        # x:  batch,seq
        embed = self.embedding(x)
        output,_ = self.gru(embed)

        batch_indexes = torch.arange(0,output.shape[0])
        lengths = (x != 0 ).sum(dim=1)
        last_hidden = output[batch_indexes,lengths-1]
        #shape : batch,hidden_size

        output = self.linear(last_hidden).squeeze(-1)
        # shape Batch,1
        return output

