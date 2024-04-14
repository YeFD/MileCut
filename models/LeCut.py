import torch as t
import torch.nn as nn

class LeCut(nn.Module):
    def __init__(self, seq_len, input_size, d_model: int=256, n_head: int=4, num_layers: int=1, dropout: float=0.4):
        super(LeCut, self).__init__()
        self.seq_len = seq_len
        self.encoding_layer = nn.LSTM(input_size=input_size, hidden_size=112 , num_layers=2, batch_first=True, bidirectional=True)
        self.position_encoding = nn.Parameter(t.randn(self.seq_len, 32), requires_grad=True)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dropout=dropout)
        self.attention_layer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decison_layer = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=1),
            nn.Softmax(dim=1)
        )
        self.shorten_layer = nn.Sequential(
            nn.Linear(in_features=768, out_features=1),
            nn.Softmax(dim=1)
            # nn.ReLU()
        )
    
    def forward(self, x):
        # x_feature = x[:,:,:5]
        # x_embd = self.shorten_layer(x[:, :, -768:])
        # x = t.cat((x_feature, x_embd), 2)
        # x = x_feature
        x = self.encoding_layer(x)[0]
        pe = self.position_encoding.expand(x.shape[0], self.seq_len, 32)
        x = t.cat((x, pe), dim=2)
        # print(x.shape)
        x = x.permute(1,0,2)
        x = self.attention_layer(x)
        x = x.permute(1,0,2)
        # x3 = self.decison_layer(t.cat((x2, x_score),2))
        x = self.decison_layer(x)
        return x


if __name__ == '__main__':
    input = t.randn(20, 100, 5)
    # a = input[:,:,:768]
    # b = input[:,:,-1:]
    # print(b.shape)
    # print(t.cat((a, b),2).shape)
    model = LeCut(seq_len=100, input_size = 5)
    result = model(input)
    print(result.size())  # (5, 300, 1)
    # print(result[:2])