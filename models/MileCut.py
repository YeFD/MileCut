import torch as t
import torch.nn as nn
import torch
from torch import Tensor
import torch.nn.functional as torch_f


class AdditiveAttention(nn.Module):
    def __init__(self, d_model: int, seq_len: int=100):
        super().__init__()
        
        self.projection = nn.Linear(in_features=d_model, out_features=d_model)
        self.query_vector_H = nn.Parameter(nn.init.xavier_uniform_(torch.empty(256, 1),
                                                                 gain=nn.init.calculate_gain('tanh')).squeeze())
        self.query_vector_L = nn.Parameter(nn.init.xavier_uniform_(torch.empty(1, seq_len),
                                                                 gain=nn.init.calculate_gain('tanh')).squeeze())

    def forward(self, inputs: Tensor, mask: Tensor):
        x = self.projection(inputs)
        attn_weight = torch.matmul(torch.tanh(x), self.query_vector_H)
        attn_weight = torch.matmul(attn_weight, self.query_vector_L)
        attn_weight.masked_fill_(~mask, 1e-30)
        attn_weight = torch_f.softmax(attn_weight, dim=1)
        output = torch.einsum('ij,ijkl->ikl', attn_weight, inputs)*inputs.shape[1]
        return output

class SimilarityNetwork(nn.Module):
    def __init__(self, input_size: int = 3, d_model: int = 256, n_head: int = 4, num_layers: int = 1, dropout: float = 0.2,
                 seq_len: int = 100):
        super(SimilarityNetwork, self).__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.encoding_layer = nn.LSTM(input_size=input_size, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dropout=dropout)
        self.attention_layer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decision_layer = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=1),
            nn.Softmax(dim=1)
        )

    def forward(self, inputs):
        inputs = self.encoding_layer(inputs)[0]
        x = inputs
        x = x.permute(1, 0, 2)
        x = self.attention_layer(x)
        x = x.permute(1, 0, 2)
        hidden_states = x

        x = self.decision_layer(x)
        return x, hidden_states

class ReasonNetwork(nn.Module):
    def __init__(self, input_size: int = 8, d_model: int = 256, n_head: int = 4, num_layers: int = 1, dropout: float = 0.2,
                 seq_len: int = 100, label_input_size=3):
        super(ReasonNetwork, self).__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.label_input_size = label_input_size
        self.encoding_layer = nn.LSTM(input_size=input_size+label_input_size, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dropout=dropout)
        self.attention_layer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decision_layer = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=1),
            nn.Softmax(dim=1)
        )

    def forward(self, inputs, labels):
        if self.label_input_size != 0:
            x = torch.concat((inputs, labels), dim=2)
        else:
            x = inputs
        x = self.encoding_layer(x)[0]
        x = x.permute(1, 0, 2)
        x = self.attention_layer(x)
        x = x.permute(1, 0, 2)
        hidden_states = x
        x = self.decision_layer(x)  
        return x, hidden_states

class MVTruncationNetwork(nn.Module):
    def __init__(self, input_size: int = 3, d_model: int = 256, n_head: int = 4, num_layers: int = 1, dropout: float = 0.2, seq_len: int = 100):
        super(MVTruncationNetwork, self).__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.position_embeddings = nn.Embedding(seq_len, d_model)
        self.encoding_layer = nn.LSTM(input_size=input_size, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dropout=dropout)
        self.attention_layer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.attn_pooling = AdditiveAttention(d_model=d_model, seq_len=seq_len)
        self.decision_layer = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=1),
            nn.Softmax(dim=1)
        )


    def forward(self, inputs, view_1_hidden_states, view_2_hidden_states, view_3_hidden_states):
        inputs = self.encoding_layer(inputs)[0]
        x = inputs
        
        position_ids = t.arange(self.seq_len, device=x.device).expand(x.shape[0], self.seq_len)
        pe = self.position_embeddings(position_ids)
        x = x + pe
        x = self.attention_layer(x)
        view_hidden_states = [view_1_hidden_states, view_2_hidden_states, view_3_hidden_states]
        view_hidden_states = torch.stack(view_hidden_states, dim=1)
        mask = torch.ones((view_hidden_states.shape[0], view_hidden_states.shape[1]), dtype=torch.bool, device=view_hidden_states.device)
        view_attn = self.attn_pooling(view_hidden_states, mask)
        x = x + view_attn
        x = self.decision_layer(x)  
        return x

class MileCut(nn.Module):
    def __init__(self, input_size: int = 3, d_model: int = 256, n_head: int = 4, num_layers: int = 1, dropout: float = 0.2, seq_len: int = 100, view_input_size: int = 3, label_input_size=3):
        super(MileCut, self).__init__()
        self.input_size = input_size
        self.view_input_size = view_input_size
        self.d_model = d_model
        self.seq_len = seq_len
        self.label_input_size = label_input_size

        self.view_1_network = SimilarityNetwork(input_size=view_input_size,
                                                  d_model=d_model,
                                                  n_head=n_head,
                                                  num_layers=num_layers,
                                                  dropout=dropout,
                                                  seq_len=seq_len)
        self.view_2_network = ReasonNetwork(input_size=view_input_size,
                                                  d_model=d_model,
                                                  n_head=n_head,
                                                  num_layers=num_layers,
                                                  dropout=dropout,
                                                  seq_len=seq_len, 
                                                  label_input_size=label_input_size)
        self.view_3_network = SimilarityNetwork(input_size=view_input_size,
                                                  d_model=d_model,
                                                  n_head=n_head,
                                                  num_layers=num_layers,
                                                  dropout=dropout,
                                                  seq_len=seq_len)
        self.truncation_network = MVTruncationNetwork(input_size=input_size,
                                                    d_model=d_model,
                                                    n_head=n_head,
                                                    num_layers=num_layers,
                                                    dropout=dropout,
                                                    seq_len=seq_len)
    def forward(self, x):
        sz = self.input_size
        sz_v = self.view_input_size
        truncation_inputs = x[:, :, :sz]
        view_1_inputs = x[:, :, sz:sz+sz_v]
        view_2_inputs = x[:, :, sz+sz_v:sz+2*sz_v]
        view_2_labels = x[:, :, sz+2*sz_v:sz+2*sz_v+self.label_input_size]
        view_3_inputs = x[:, :, sz+2*sz_v+self.label_input_size:]
        view_1_output, view_1_hidden_states = self.view_1_network(view_1_inputs)
        view_2_output, view_2_hidden_states = self.view_2_network(view_2_inputs, view_2_labels)
        view_3_output, view_3_hidden_states = self.view_3_network(view_3_inputs)
        truncation_output = self.truncation_network(truncation_inputs, view_1_hidden_states, view_2_hidden_states, view_3_hidden_states)
        return truncation_output, view_1_output, view_2_output, view_3_output

if __name__ == '__main__':
    input = t.randn(20, 100, 18)
    model = MileCut(seq_len=100, input_size=6, view_input_size=3)
    result, _, _, _ = model(input)
    print(result.size())