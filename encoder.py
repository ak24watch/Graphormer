import torch
import torch.nn as nn

class FeedForwardNetwork(nn.Module):
    """
    Feed Forward Network used in the encoder.

    Args:
        hidden_size (int): Size of the hidden layer.
        ffn_size (int): Size of the feed-forward layer.
        encoder_droput (float): Dropout rate.
    """
    def __init__(self, hidden_size, ffn_size, encoder_dropout):  # corrected typo here
        super(FeedForwardNetwork, self).__init__()
        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)
        self.fnn_dropout = nn.Dropout(encoder_dropout)  # corrected typo here

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
        hidden_size (int): Size of the hidden layer.
        attention_drop (float): Dropout rate for attention.
        num_heads (int): Number of attention heads.
    """
    def __init__(self, hidden_size, attention_drop, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.att_size = hidden_size // num_heads
        self.scale = self.att_size**-0.5
        self.linear_q = nn.Linear(hidden_size, num_heads * self.att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * self.att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * self.att_size)
        self.att_dropout = nn.Dropout(attention_drop)
        self.output_layer = nn.Linear(num_heads * self.att_size, hidden_size)

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
        q = self.linear_q(h).view(batch_size, seq_len, self.num_heads, self.att_size)
        k = self.linear_k(h).view(batch_size, seq_len, self.num_heads, self.att_size)
        v = self.linear_v(h).view(batch_size, seq_len, self.num_heads, self.att_size)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 3, 1)
        v = v.permute(0, 2, 1, 3)
        scores = torch.matmul(q, k) * self.scale
        att_bias = att_bias.permute(0, 3, 1, 2)
        if att_bias is not None:
            scores = scores + att_bias
        if mask is not None:
            mask = mask.unsqueeze(1)
            mask = mask.repeat(1, self.num_heads, 1, 1)
        if mask is not None:
            scores = scores.masked_fill(
                mask.to(torch.bool), float(10e-15)
            )
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.att_dropout(attention_weights)
        output = torch.matmul(
            attention_weights, v
        )
        output = output.permute(
            0, 2, 1, 3
        ).contiguous()
        output = output.view(batch_size, seq_len, self.num_heads * self.att_size)
        output = self.output_layer(output)
        return output

class Encoder(nn.Module):
    """
    Encoder layer consisting of multi-head attention and feed-forward network.

    Args:
        hidden_size (int): Size of the hidden layer.
        ffn_out_size (int): Size of the feed-forward layer.
        attention_dropout (float): Dropout rate for attention.
        num_heads (int): Number of attention heads.
    """
    def __init__(self, hidden_size, ffn_out_size, attention_dropout, num_heads):
        super().__init__()
        self.self_att_norm = nn.LayerNorm(hidden_size)
        self.self_att = Attention(hidden_size, attention_dropout, num_heads)
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(
            hidden_size, ffn_out_size, encoder_dropout=0.1  # corrected typo here
        )

    def forward(self, x, att_bias=None, att_mask=None):
        """
        Forward pass for the encoder layer.

        Args:
            x (Tensor): Input tensor.
            att_bias (Tensor, optional): Attention bias tensor.
            att_mask (Tensor, optional): Attention mask tensor.

        Returns:
            Tensor: Output tensor.
        """
        h = self.self_att_norm(x)
        h = self.self_att(
            h, att_bias=att_bias, mask=att_mask
        )
        y = h + x
        h = self.ffn_norm(y)
        h = self.ffn(h)
        y = h + y
        return y
