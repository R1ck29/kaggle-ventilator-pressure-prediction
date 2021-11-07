import torch
import torch.nn as nn
import torch.nn.init as init

# ====================================================
# Model
# ====================================================
class CustomModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.hidden_size = cfg.hidden_size
        self.r_emb = nn.Embedding(3, 2, padding_idx=0)
        self.c_emb = nn.Embedding(3, 2, padding_idx=0)
        self.seq_emb = nn.Sequential(
            nn.Linear(4 + len(cfg.cont_seq_cols), self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        if self.cfg.model_name == 'rnn':
            self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, 
                            dropout=0.2, batch_first=True, bidirectional=True)
            
            self.head = nn.Sequential(
                nn.Linear(self.hidden_size * 2, self.hidden_size * 2),
                nn.LayerNorm(self.hidden_size * 2),
                nn.ReLU(),
                nn.Dropout(0.),
                nn.Linear(self.hidden_size * 2, 1),
            )
        elif self.cfg.model_name == 'rnn_v2':
            self.lstm1 = nn.LSTM(self.hidden_size, self.hidden_size//2, dropout=0.1, batch_first=True, bidirectional=True)
            self.lstm2 = nn.LSTM(self.hidden_size//2 * 2, self.hidden_size//4, dropout=0.1, batch_first=True, bidirectional=True)
            self.lstm3 = nn.LSTM(self.hidden_size//4 * 2, self.hidden_size//8, dropout=0.1, batch_first=True, bidirectional=True)

            self.head = nn.Sequential(
                nn.Linear(self.hidden_size//8 * 2, self.hidden_size//8 * 2),
                nn.LayerNorm(self.hidden_size//8 * 2),
                nn.GELU(),
                nn.Dropout(0.),
                nn.Linear(self.hidden_size//8 * 2, 1),
            )

        for n, m in self.named_modules():
            if isinstance(m, nn.LSTM):
                print(f'init {m}')
                for param in m.parameters():
                    if len(param.shape) >= 2:
                        nn.init.orthogonal_(param.data)
                    else:
                        nn.init.normal_(param.data)
            elif isinstance(m, nn.GRU):
                print(f"init {m}")
                for param in m.parameters():
                    if len(param.shape) >= 2:
                        init.orthogonal_(param.data)
                    else:
                        init.normal_(param.data)

    def forward(self, cate_seq_x, cont_seq_x):
        bs = cont_seq_x.size(0)
        r_emb = self.r_emb(cate_seq_x[:,:,0]).view(bs, 80, -1)
        c_emb = self.c_emb(cate_seq_x[:,:,1]).view(bs, 80, -1)
        seq_x = torch.cat((r_emb, c_emb, cont_seq_x), 2)
        seq_emb = self.seq_emb(seq_x)
        if self.cfg.model_name == 'rnn':
            seq_emb, _ = self.lstm(seq_emb)
        elif self.cfg.model_name == 'rnn_v2':
            seq_emb, _ = self.lstm1(seq_emb)
            seq_emb, _ = self.lstm2(seq_emb)
            seq_emb, _ = self.lstm3(seq_emb)
        output = self.head(seq_emb).view(bs, -1)
        return output


class CustomModel_v2(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.hidden_size = cfg.hidden_size
        self.act_function = nn.GELU()
        # else:
        #     self.act_function = nn.SELU()
        self.seq_emb = nn.Sequential(
            nn.Linear(len(cfg.cont_seq_cols), self.hidden_size),
            nn.LayerNorm(self.hidden_size),    
            self.act_function,
            nn.Dropout(0.1),
        )
        self.lstm1 = nn.LSTM(self.hidden_size, self.hidden_size//2, dropout=0.1, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(self.hidden_size//2 * 2, self.hidden_size//4, dropout=0.1, batch_first=True, bidirectional=True)
        self.lstm3 = nn.LSTM(self.hidden_size//4 * 2, self.hidden_size//8, dropout=0.1, batch_first=True, bidirectional=True)
        # 512 128
        if self.hidden_size == 1024:
            self.lstm4 = nn.LSTM(self.hidden_size//8 * 2, self.hidden_size//16, dropout=0.1, batch_first=True, bidirectional=True)
            # in=256 out=64
            self.head = nn.Sequential(
                # nn.Linear(self.hidden_size//8 * 2, self.hidden_size//8 * 2),
                nn.LayerNorm(self.hidden_size//16 * 2),
                self.act_function,
                #nn.Dropout(0.),
                nn.Linear(self.hidden_size//16 * 2, 1),
            )
        elif self.hidden_size == 2048:
            self.lstm4 = nn.LSTM(self.hidden_size//8 * 2, self.hidden_size//16, dropout=0.1, batch_first=True, bidirectional=True)
            self.lstm5 = nn.LSTM(self.hidden_size//16 * 2, self.hidden_size//32, dropout=0.1, batch_first=True, bidirectional=True)
            # in=256 out=64
            self.head = nn.Sequential(
                nn.LayerNorm(self.hidden_size//32 * 2),
                self.act_function,
                #nn.Dropout(0.),
                nn.Linear(self.hidden_size//32 * 2, 1),
            )
        else:
            self.head = nn.Sequential(
                # nn.Linear(self.hidden_size//8 * 2, self.hidden_size//8 * 2),
                nn.LayerNorm(self.hidden_size//8 * 2),
                self.act_function,
                #nn.Dropout(0.),
                nn.Linear(self.hidden_size//8 * 2, 1),
            )
            
        if cfg.experiment_name == 'config16' or cfg.experiment_name == 'config20':
            print('Applying xavier_uniform.')
            def _weights_init(m):
                if isinstance(m, (nn.LSTM, nn.GRU)):
                    nn.init.xavier_uniform_(m.weight_ih_l0)
                    nn.init.orthogonal_(m.weight_hh_l0)
                    nn.init.xavier_uniform_(m.weight_ih_l0_reverse)
                    nn.init.orthogonal_(m.weight_hh_l0_reverse)
                
            self.apply(_weights_init)
        elif cfg.experiment_name == 'config24' or cfg.experiment_name == 'config25':
            self._reinitialize()
        else:
            for n, m in self.named_modules():
                if isinstance(m, nn.LSTM):
                    print(f'init {m}')
                    for param in m.parameters():
                        if len(param.shape) >= 2:
                            nn.init.orthogonal_(param.data)
                        else:
                            nn.init.normal_(param.data)
                elif isinstance(m, nn.GRU):
                    print(f"init {m}")
                    for param in m.parameters():
                        if len(param.shape) >= 2:
                            init.orthogonal_(param.data)
                        else:
                            init.normal_(param.data)

        if cfg.experiment_name == 'config21' or cfg.experiment_name == 'config23' or cfg.experiment_name == 'config26':
            print('Applying xavier_uniform.overwriting...')
            def _weights_init(m):
                if isinstance(m, (nn.LSTM, nn.GRU)):
                    nn.init.xavier_uniform_(m.weight_ih_l0)
                    nn.init.orthogonal_(m.weight_hh_l0)
                    nn.init.xavier_uniform_(m.weight_ih_l0_reverse)
                    nn.init.orthogonal_(m.weight_hh_l0_reverse) 
            self.apply(_weights_init)

    def _reinitialize(self):
        """
        Tensorflow/Keras-like initialization
        """
        print('Tensorflow/Keras-like initialization')
        for name, p in self.named_parameters():
            if 'lstm' in name:
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(p.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(p.data)
                elif 'bias_ih' in name:
                    p.data.fill_(0)
                    # Set forget-gate bias to 1
                    n = p.size(0)
                    p.data[(n // 4):(n // 2)].fill_(1)
                elif 'bias_hh' in name:
                    p.data.fill_(0)
            elif 'fc' in name:
                if 'weight' in name:
                    nn.init.xavier_uniform_(p.data)
                elif 'bias' in name:
                    p.data.fill_(0) 


    def forward(self, cont_seq_x):
        bs = cont_seq_x.size(0)
        seq_emb = self.seq_emb(cont_seq_x)
        seq_emb, _ = self.lstm1(seq_emb)
        seq_emb, _ = self.lstm2(seq_emb)
        seq_emb, _ = self.lstm3(seq_emb)
        if self.hidden_size == 1024:
            seq_emb, _ = self.lstm4(seq_emb)
        if self.hidden_size == 2048:
            seq_emb, _ = self.lstm4(seq_emb)
            seq_emb, _ = self.lstm5(seq_emb)
        output = self.head(seq_emb)#.view(bs, -1)
        return output