import torch
import torch.nn as nn


class DynamicPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(DynamicPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.position_encoding = nn.Parameter(torch.zeros(1, max_len, d_model))

    def forward(self, x):
        seq_len = x.size(1)
        pos_encoding = self.position_encoding[:, :seq_len, :]
        return x + pos_encoding

class TransformerEncoderLayerWithMixedAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, local_window_size=5, dropout=0.1):
        super(TransformerEncoderLayerWithMixedAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.position_encoding = DynamicPositionalEncoding(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.local_window_size = local_window_size

    def forward(self, x):
        x = self.position_encoding(x)
        seq_len, batch_size, _ = x.size()
        device = x.device
        attn_mask = torch.zeros(seq_len, seq_len, device=device)

        for i in range(seq_len):
            start = max(0, i - self.local_window_size)
            end = min(seq_len, i + self.local_window_size + 1)
            attn_mask[i, :start] = float('-inf')
            attn_mask[i, end:] = float('-inf')

        sparse_step = self.local_window_size * 2
        for i in range(0, seq_len, sparse_step):
            attn_mask[:, i] = 0

        attn_output, _ = self.self_attention(x, x, x, attn_mask=attn_mask)
        x = self.layernorm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.layernorm2(x + self.dropout(ff_output))
        return x

class TransformerDecoderLayerWithMixedAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, local_window_size=5, dropout=0.1):
        super(TransformerDecoderLayerWithMixedAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.layernorm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.local_window_size = local_window_size

    def forward(self, x, encoder_output):
        x = self.apply_mixed_attention(x, x, x)
        x = self.apply_mixed_attention(x, encoder_output, encoder_output, cross_attention=True)
        ff_output = self.feed_forward(x)
        x = self.layernorm3(x + self.dropout(ff_output))
        return x

    def apply_mixed_attention(self, query, key, value, cross_attention=False):
        seq_len_q, batch_size_q, _ = query.size()
        seq_len_k, batch_size_k, _ = key.size()
        device = query.device
        attn_mask = torch.zeros(seq_len_q, seq_len_k, device=device)

        for i in range(seq_len_q):
            start = max(0, i - self.local_window_size)
            end = min(seq_len_k, i + self.local_window_size + 1)
            attn_mask[i, :start] = float('-inf')
            attn_mask[i, end:] = float('-inf')

        sparse_step = self.local_window_size * 2
        for i in range(0, seq_len_k, sparse_step):
            attn_mask[:, i] = 0

        attention_layer = self.self_attention if not cross_attention else self.cross_attention
        attn_output, _ = attention_layer(query, key, value, attn_mask=attn_mask)
        layernorm = self.layernorm1 if not cross_attention else self.layernorm2
        x = layernorm(query + self.dropout(attn_output))
        return x

class TransformerRecommenderWithMixedAttention(nn.Module):
    def __init__(self, input_dim, maxlen, embed_dim, num_heads, ff_dim, num_encoders=2, num_decoders=2, num_layers=2, dropout=0.1):
        super(TransformerRecommenderWithMixedAttention, self).__init__()
        self.embedding = nn.Embedding(input_dim, embed_dim)
        self.conv1d = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)
        self.encoders = nn.ModuleList([
            nn.ModuleList([
                TransformerEncoderLayerWithMixedAttention(embed_dim, num_heads, ff_dim, dropout=dropout)
                for _ in range(num_layers)
            ]) for _ in range(num_encoders)
        ])
        self.decoders = nn.ModuleList([
            nn.ModuleList([
                TransformerDecoderLayerWithMixedAttention(embed_dim, num_heads, ff_dim, dropout=dropout)
                for _ in range(num_layers)
            ]) for _ in range(num_decoders)
        ])
        self.pooling = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(embed_dim * num_decoders, 20)
        self.fc2 = nn.Linear(20, 2)
        
        self.adjust_encoder_output = nn.Linear(embed_dim * num_encoders, embed_dim)

    def forward(self, x, target_seq):
        # 编码器部分
        x = self.embedding(x.long())
        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        x = x.permute(2, 0, 1)
        encoder_outputs = []
        for encoder_layers in self.encoders:
            x_enc = x
            for encoder_layer in encoder_layers:
                x_enc = encoder_layer(x_enc)
            encoder_outputs.append(x_enc)
        combined_encoder_output = torch.cat(encoder_outputs, dim=-1)
        combined_encoder_output = self.adjust_encoder_output(combined_encoder_output)

        # 解码器部分
        target_seq = self.embedding(target_seq.long())
        target_seq = target_seq.permute(0, 2, 1)
        target_seq = self.conv1d(target_seq)
        target_seq = target_seq.permute(2, 0, 1)
        x_dec = target_seq

        enc_seq_len = combined_encoder_output.size(0)
        dec_seq_len = x_dec.size(0)
        if enc_seq_len != dec_seq_len:
            min_seq_len = min(enc_seq_len, dec_seq_len)
            combined_encoder_output = combined_encoder_output[:min_seq_len, :, :]
            x_dec = x_dec[:min_seq_len, :, :]

        # 收集解码器输出
        decoder_outputs = []
        for decoder_layers in self.decoders:
            x_dec_temp = x_dec
            for decoder_layer in decoder_layers:
                x_dec_temp = decoder_layer(x_dec_temp, combined_encoder_output)
            decoder_outputs.append(x_dec_temp)

        # 合并解码器输出
        combined_decoder_output = torch.cat(decoder_outputs, dim=-1)

        # 池化和预测
        x_dec = combined_decoder_output.permute(1, 2, 0)
        pooled_output = self.pooling(x_dec).squeeze(-1)
        x = torch.relu(self.fc1(pooled_output))
        x = self.fc2(x)
        return x

