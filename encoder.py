import torch
import torch.nn as nn


class FeedForwardNetwork(nn.Module):
    """
    Feed Forward Network used in the encoder.

    Args:
        cfg (object): Configuration object containing model parameters.
    """

    def __init__(self, cfg):
        super(FeedForwardNetwork, self).__init__()
        self.layer1 = nn.Linear(
            2 * cfg.d_model if cfg.concat_pos_emb else cfg.d_model, cfg.d_ffn
        )
        self.gelu = cfg.ffn_activation
        self.layer2 = nn.Linear(
            cfg.d_ffn, 2 * cfg.d_model if cfg.concat_pos_emb else cfg.d_model
        )
        self.fnn_dropout = nn.Dropout(cfg.ffn_dropout)

    def forward(self, x):
        """
        Forward pass for the feed-forward network.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        x = self.fnn_dropout(x)
        return x


class Attention(nn.Module):
    """
    Multi-head Attention mechanism.

    Args:
        cfg (object): Configuration object containing model parameters.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.scale = cfg.d_head**-0.5
        self.linear_q = nn.Linear(
            2 * cfg.d_model if cfg.concat_pos_emb else cfg.d_model,
            cfg.n_heads * cfg.d_head,
            bias=False,
        )
        self.linear_k = nn.Linear(
            2 * cfg.d_model if cfg.concat_pos_emb else cfg.d_model,
            cfg.n_heads * cfg.d_head,
            bias=False,
        )
        self.linear_v = nn.Linear(
            2 * cfg.d_model if cfg.concat_pos_emb else cfg.d_model,
            cfg.n_heads * cfg.d_head,
            bias=False,
        )
        self.att_dropout = nn.Dropout(cfg.attention_dropout)
        self.output_layer = nn.Linear(
            cfg.n_heads * cfg.d_head,
            2 * cfg.d_model if cfg.concat_pos_emb else cfg.d_model,
            bias=False,
        )

    def forward(self, h, att_bias=None, mask=None):
        """
        Forward pass for the attention mechanism.

        Args:
            h (Tensor): Input tensor.
            att_bias (Tensor, optional): Attention bias tensor.
            mask (Tensor, optional): Attention mask tensor.

        Returns:
            Tensor: Output tensor.
        """
        batch_size, seq_len = h.shape[:2]
        q = self.linear_q(h).view(
            batch_size, seq_len, self.cfg.n_heads, self.cfg.d_head
        )
        k = self.linear_k(h).view(
            batch_size, seq_len, self.cfg.n_heads, self.cfg.d_head
        )
        v = self.linear_v(h).view(
            batch_size, seq_len, self.cfg.n_heads, self.cfg.d_head
        )
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 3, 1)
        v = v.permute(0, 2, 1, 3)
        scores = torch.matmul(q, k) * self.scale

        if att_bias is not None:
            att_bias = att_bias.permute(0, 3, 1, 2)
            scores = scores + att_bias
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask.to(torch.bool), float(10e-5))
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.att_dropout(attention_weights)
        output = torch.matmul(attention_weights, v)
        output = output.permute(0, 2, 1, 3).contiguous()
        output = output.view(batch_size, seq_len, self.cfg.n_heads * self.cfg.d_head)
        output = self.output_layer(output)
        return output


class Encoder(nn.Module):
    """
    Encoder layer consisting of multi-head attention and feed-forward network.

    Args:
        cfg (object): Configuration object containing model parameters.
    """

    def __init__(self, cfg):
        super().__init__()
        self.self_att_norm = nn.LayerNorm(
            2 * cfg.d_model if cfg.concat_pos_emb else cfg.d_model
        )
        self.self_att = Attention(cfg)
        self.ffn_norm = nn.LayerNorm(
            2 * cfg.d_model if cfg.concat_pos_emb else cfg.d_model
        )
        self.ffn = FeedForwardNetwork(cfg)

    def forward(self, residual_pre, att_bias=None, att_mask=None):
        """
        Forward pass for the encoder layer.

        Args:
            residual_pre (Tensor): Input tensor.
            att_bias (Tensor, optional): Attention bias tensor.
            att_mask (Tensor, optional): Attention mask tensor.

        Returns:
            Tensor: Output tensor.
        """
        normalized_residual_pre = self.self_att_norm(residual_pre)
        att_out = self.self_att(
            normalized_residual_pre, att_bias=att_bias, mask=att_mask
        )
        residual_mid = residual_pre + att_out
        normalized_residual_mid = self.ffn_norm(residual_mid)
        ffn_out = self.ffn(normalized_residual_mid)
        residual_post = residual_mid + ffn_out
        return residual_post
