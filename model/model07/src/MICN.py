import torch
import torch.nn as nn
from shared_components.layers.Embed import DataEmbedding
from shared_components.layers.Autoformer_EncDec import series_decomp, series_decomp_multi
import torch.nn.functional as F


class CrossStockAttention(nn.Module):
    """横截面（同一交易日内股票间）自注意力。

    让每只股票的表示可以关注当天其它所有股票，从而学习“谁比谁强”的相对关系。
    输入/输出形状均为 [batch, num_stocks, d_model]，对 num_stocks 数量无固定要求。
    移植自 model01 的同名模块，是其跑赢其它单模型的关键结构。
    """

    def __init__(self, d_model, nhead, dropout=0.1):
        super(CrossStockAttention, self).__init__()
        self.cross_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, stock_features):
        # stock_features: [batch, num_stocks, d_model]
        attended, _ = self.cross_attention(stock_features, stock_features, stock_features)
        return self.norm(stock_features + self.dropout(attended))


class MIC(nn.Module):
    def __init__(self, feature_size=512, n_heads=8, dropout=0.05, decomp_kernel=[32], conv_kernel=[24],
                 isometric_kernel=[18, 6], device='cuda'):
        super(MIC, self).__init__()
        self.conv_kernel = conv_kernel
        self.device = device

        self.isometric_conv = nn.ModuleList([nn.Conv1d(in_channels=feature_size, out_channels=feature_size,
                                                       kernel_size=i, padding=0, stride=1)
                                             for i in isometric_kernel])
        self.conv = nn.ModuleList([nn.Conv1d(in_channels=feature_size, out_channels=feature_size,
                                             kernel_size=i, padding=i // 2, stride=i)
                                   for i in conv_kernel])
        self.conv_trans = nn.ModuleList([nn.ConvTranspose1d(in_channels=feature_size, out_channels=feature_size,
                                                            kernel_size=i, padding=0, stride=i)
                                         for i in conv_kernel])
        self.decomp = nn.ModuleList([series_decomp(k) for k in decomp_kernel])
        self.merge = torch.nn.Conv2d(in_channels=feature_size, out_channels=feature_size,
                                     kernel_size=(len(self.conv_kernel), 1))
        self.conv1 = nn.Conv1d(in_channels=feature_size, out_channels=feature_size * 4, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=feature_size * 4, out_channels=feature_size, kernel_size=1)
        self.norm1 = nn.LayerNorm(feature_size)
        self.norm2 = nn.LayerNorm(feature_size)
        self.norm = torch.nn.LayerNorm(feature_size)
        self.act = torch.nn.Tanh()
        self.drop = torch.nn.Dropout(0.05)

    def conv_trans_conv(self, input, conv1d, conv1d_trans, isometric):
        batch, seq_len, channel = input.shape
        x = input.permute(0, 2, 1)
        x1 = self.drop(self.act(conv1d(x)))
        x = x1
        zeros = torch.zeros((x.shape[0], x.shape[1], x.shape[2] - 1), device=self.device)
        x = torch.cat((zeros, x), dim=-1)
        x = self.drop(self.act(isometric(x)))
        x = self.norm((x + x1).permute(0, 2, 1)).permute(0, 2, 1)
        x = self.drop(self.act(conv1d_trans(x)))
        x = x[:, :, :seq_len]  # truncate
        x = self.norm(x.permute(0, 2, 1) + input)
        return x

    def forward(self, src):
        self.device = src.device
        multi = []
        for i in range(len(self.conv_kernel)):
            src_out, _ = self.decomp[i](src)
            src_out = self.conv_trans_conv(src_out, self.conv[i], self.conv_trans[i], self.isometric_conv[i])
            multi.append(src_out)
        mg = torch.tensor([], device=self.device)
        for i in range(len(self.conv_kernel)):
            mg = torch.cat((mg, multi[i].unsqueeze(1).to(self.device)), dim=1)
        mg = self.merge(mg.permute(0, 3, 1, 2)).squeeze(-2).permute(0, 2, 1)
        y = self.norm1(mg)
        y = self.conv2(self.conv1(y.transpose(-1, 1))).transpose(-1, 1)
        return self.norm2(mg + y)


class SeasonalPrediction(nn.Module):
    def __init__(self, embedding_size=512, n_heads=8, dropout=0.05, d_layers=1, decomp_kernel=[32], c_out=1,
                 conv_kernel=[2, 4], isometric_kernel=[18, 6], device='cuda'):
        super(SeasonalPrediction, self).__init__()
        self.mic = nn.ModuleList([MIC(feature_size=embedding_size, n_heads=n_heads,
                                      decomp_kernel=decomp_kernel, conv_kernel=conv_kernel,
                                      isometric_kernel=isometric_kernel, device=device)
                                  for i in range(d_layers)])
        self.projection = nn.Linear(embedding_size, c_out)

    def forward(self, dec):
        for mic_layer in self.mic:
            dec = mic_layer(dec)
        return self.projection(dec)


class Model(nn.Module):
    def __init__(self, configs, conv_kernel=[12, 16]):
        super(Model, self).__init__()
        decomp_kernel = []
        isometric_kernel = []
        for ii in conv_kernel:
            if ii % 2 == 0:
                decomp_kernel.append(ii + 1)
                isometric_kernel.append((configs.seq_len + configs.pred_len + ii) // ii)
            else:
                decomp_kernel.append(ii)
                isometric_kernel.append((configs.seq_len + configs.pred_len + ii - 1) // ii)

        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len

        self.decomp_multi = series_decomp_multi(decomp_kernel)
        self.dec_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)

        # 【修复】：将写死的 cuda:0 换成动态设备检测
        current_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.conv_trans = SeasonalPrediction(embedding_size=configs.d_model, n_heads=configs.n_heads,
                                             dropout=configs.dropout, d_layers=configs.d_layers,
                                             decomp_kernel=decomp_kernel, c_out=configs.c_out,
                                             conv_kernel=conv_kernel, isometric_kernel=isometric_kernel,
                                             device=current_device)

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.regression = nn.Linear(configs.seq_len, configs.pred_len)
            self.regression.weight = nn.Parameter(
                (1 / configs.pred_len) * torch.ones([configs.pred_len, configs.seq_len]), requires_grad=True)
        if self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            # 横截面排序头：时间维聚合 -> 投影 -> 股票间注意力 -> 打分
            self.cs_proj = nn.Linear(configs.c_out, configs.d_model)
            self.cross_stock_attention = CrossStockAttention(configs.d_model, configs.n_heads, configs.dropout)
            self.score_head = nn.Sequential(
                nn.Linear(configs.d_model, configs.d_model // 2),
                nn.ReLU(),
                nn.Dropout(configs.dropout),
                nn.Linear(configs.d_model // 2, configs.num_class),
            )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        seasonal_init_enc, trend = self.decomp_multi(x_enc)
        trend = self.regression(trend.permute(0, 2, 1)).permute(0, 2, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)
        seasonal_init_dec = torch.cat([seasonal_init_enc[:, -self.seq_len:, :], zeros], dim=1)
        dec_out = self.dec_embedding(seasonal_init_dec, x_mark_dec)
        dec_out = self.conv_trans(dec_out)
        dec_out = dec_out[:, -self.pred_len:, :] + trend[:, -self.pred_len:, :]
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        seasonal_init_enc, trend = self.decomp_multi(x_enc)
        dec_out = self.dec_embedding(seasonal_init_enc, x_mark_dec)
        dec_out = self.conv_trans(dec_out)
        dec_out = dec_out + trend
        return dec_out

    def anomaly_detection(self, x_enc):
        seasonal_init_enc, trend = self.decomp_multi(x_enc)
        dec_out = self.dec_embedding(seasonal_init_enc, None)
        dec_out = self.conv_trans(dec_out)
        dec_out = dec_out + trend
        return dec_out

    def classification(self, x_enc, x_mark_enc=None):  # 增加默认参数 =None
        # x_enc: [N, L, F]，其中 N 为当天参与排序的股票数（作为 batch 维）
        seasonal_init_enc, trend = self.decomp_multi(x_enc)
        dec_out = self.dec_embedding(seasonal_init_enc, None)
        dec_out = self.conv_trans(dec_out)
        dec_out = dec_out + trend                       # [N, L, c_out]

        output = self.act(dec_out)
        output = self.dropout(output)                   # [N, L, c_out]

        # 时间维聚合，得到每只股票的表示
        stock_repr = output.mean(dim=1)                 # [N, c_out]
        stock_repr = self.cs_proj(stock_repr)           # [N, d_model]

        # 横截面：把 N 只股票作为序列，做股票间自注意力
        stock_repr = stock_repr.unsqueeze(0)            # [1, N, d_model]
        stock_repr = self.cross_stock_attention(stock_repr)

        scores = self.score_head(stock_repr)            # [1, N, num_class]
        return scores.squeeze(0)                        # [N, num_class]

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out
        return None