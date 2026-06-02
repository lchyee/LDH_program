import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class ResBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1, bias=True):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=bias)
        self.fc3 = nn.Linear(input_dim, output_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.ln = LayerNorm(output_dim, bias=bias)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + self.fc3(x)
        out = self.ln(out)
        return out


class Model(nn.Module):
    def __init__(self, configs, bias=True, feature_encode_dim=2):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.hidden_dim = configs.d_model
        self.res_hidden = configs.d_model
        self.encoder_num = configs.e_layers
        self.decoder_num = configs.d_layers
        self.freq = configs.freq
        self.feature_encode_dim = feature_encode_dim
        self.decode_dim = configs.c_out
        self.temporalDecoderHidden = configs.d_ff
        dropout = configs.dropout

        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        self.feature_dim = freq_map[self.freq]

        # ========================================================
        # 【核心修复】：为 TiDE 添加原本不支持的 Classification 结构
        # ========================================================
        if self.task_name == 'classification':
            flatten_dim = self.seq_len * configs.enc_in
            self.encoders = nn.Sequential(
                ResBlock(flatten_dim, self.res_hidden, self.hidden_dim, dropout, bias),
                *([ResBlock(self.hidden_dim, self.res_hidden, self.hidden_dim, dropout, bias)] * (self.encoder_num - 1))
            )
            self.projection = nn.Linear(self.hidden_dim, configs.num_class)
        else:
            # 原版逻辑保留
            flatten_dim = self.seq_len + (self.seq_len + self.pred_len) * self.feature_encode_dim
            self.feature_encoder = ResBlock(self.feature_dim, self.res_hidden, self.feature_encode_dim, dropout, bias)
            self.encoders = nn.Sequential(ResBlock(flatten_dim, self.res_hidden, self.hidden_dim, dropout, bias), *(
                        [ResBlock(self.hidden_dim, self.res_hidden, self.hidden_dim, dropout, bias)] * (
                            self.encoder_num - 1)))
            # ... 其它任务网络省略，因为用不到 ...

    # ========================================================
    # 【新增方法】：打分逻辑
    # ========================================================
    def classification(self, x_enc, x_mark_enc=None):
        # 1. 将 [Batch, Seq_len, Features] 展平为 [Batch, Seq_len * Features]
        x = x_enc.reshape(x_enc.shape[0], -1)
        # 2. 通过 TiDE 核心的 MLP 残差网络
        hidden = self.encoders(x)
        # 3. 映射到最终得分
        output = self.projection(hidden)
        return output

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, batch_y_mark=None, mask=None):
        if self.task_name == 'classification':
            return self.classification(x_enc, x_mark_enc)
        # 原版抛出异常的逻辑已替换
        return None