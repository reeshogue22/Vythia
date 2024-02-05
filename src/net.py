# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from einops import rearrange

# class Conv(nn.Module):
#     def __init__(self, in_channel, out_channel, kernel_size, stride, padding, bias=False):
#         super(Conv, self).__init__()
#         self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, bias=bias)

#     def forward(self, x):
#         # Reshape input tensor to combine the spatial and time dimensions
#         b, c, h, w, t = x.size()
#         x_reshaped = x.permute(0, 4, 1, 2, 3).contiguous()
#         x_reshaped = rearrange(x_reshaped, 'b t c h w -> (b t) c h w')
        
#         # Apply convolution in a batched manner
#         y_reshaped = self.conv(x_reshaped.contiguous())

#         # Reshape the output back to the original shape
#         y = rearrange(y_reshaped, '(b t) c h w -> b t c h w', t=t)
#         y = y.permute(0, 2, 3, 4, 1)
        
#         return y

# class Norm(nn.Module):
#     def __init__(self, in_channel):
#         super(Norm, self).__init__()
#         self.norm = nn.GroupNorm(1, in_channel)
#     def forward(self, x):
#         # Reshape input tensor to combine the spatial and time dimensions
#         b, c, h, w, t = x.size()
#         x_reshaped = x.permute(0, 4, 1, 2, 3).contiguous()
#         x_reshaped = rearrange(x_reshaped, 'b t c h w -> (b t) c h w')

#         # Apply normalization a batched manner
#         y_reshaped = self.norm(x_reshaped)

#         # Reshape the output back to the original shape
#         y = rearrange(y_reshaped, '(b t) c h w -> b t c h w', t=t)
#         y = y.permute(0, 2, 3, 4, 1)

#         return y
    
# class Attention(nn.Module):
#     def __init__(self, dmodel, nheads):
#         super(Attention, self).__init__()
#         self.dmodel = dmodel
#         self.nheads = nheads

#         # Temporal 2D convolutions for Q, K, V
#         self.conv_q = Conv(dmodel, dmodel, 3, 1, 1)
#         self.conv_k = Conv(dmodel, dmodel//nheads, 1, 1, 0)
#         self.conv_v = Conv(dmodel, dmodel//nheads, 1, 1, 0)
#         self.conv_r = Conv(dmodel, dmodel, 1, 1, 0)

#         self.conv_proj = Conv(dmodel, dmodel, 1, 1, 0)

#     def forward(self, x):
#         # Apply temporal 2D convolutions for Q, K, V
#         q = self.conv_q(x)
#         k = self.conv_k(x)
#         v = self.conv_v(x)
#         r = self.conv_r(x)

#         # Flatten along spatial dimensions for attention
#         q_flat = rearrange(q, 'b c h w t -> b t (c h w)')
#         k_flat = rearrange(k, 'b c h w t -> b t (c h w)')
#         v_flat = rearrange(v, 'b c h w t -> b t (c h w)')

#         q_headed = rearrange(q_flat, 'b t (head d) -> b head t d', head=self.nheads)
#         k_headed = rearrange(k_flat, 'b t (head d) -> b head t d', head=1)
#         v_headed = rearrange(v_flat, 'b t (head d) -> b head t d', head=1)

#         #All you need.
#         attention_output = F.scaled_dot_product_attention(q_headed, k_headed, v_headed, is_causal=True)

#         attention_output = rearrange(attention_output, 'b head t d -> b t (head d)')

#         # Reshape attention output back to (b, c, h, w, t)
#         attention_output = rearrange(attention_output, 'b t (c h w) -> b c h w t', h=q.size(2), w=q.size(3), c=q.shape[1])

#         attention_output = attention_output * torch.sigmoid(r)

#         attention_output = self.conv_proj(attention_output)

#         return q

# class FFN(nn.Module):

#     def __init__(self, dmodel):
#         super(FFN, self).__init__()
#         self.conv1 = Conv(dmodel, dmodel*4, 1, 1, 0, bias=False)
#         self.conv2 = Conv(dmodel*2, dmodel, 1, 1, 0, bias=False)
#     def forward(self, x):
#         x1, x2 = self.conv1(x).chunk(2, 1)
#         x = x1 * x2
#         x = self.conv2(x)
#         return x
    
# class ConvTransformerBlock(nn.Module):


#     def __init__(self, dmodel, nheads, dropout=0.1):
#         super(ConvTransformerBlock, self).__init__()
#         self.attention = Attention(dmodel, nheads)
#         self.ffn = FFN(dmodel)
#         self.norm = Norm(dmodel)
#         self.norm2 = Norm(dmodel)
#         self.gelu = nn.GELU()
#     def forward(self, x):
#         x = self.gelu(self.attention(self.norm(x)) + x)
#         x = self.gelu(self.ffn(self.norm2(x)) + x)
#         return x

# class ConvTransformer(nn.Module):
#     def __init__(self, chan, outchan, dmodel, nheads, num_blocks):
#         super(ConvTransformer, self).__init__()
#         self.blocks = nn.ModuleList([ConvTransformerBlock(dmodel, nheads) for _ in range(num_blocks)])
#         self.conv_in = Conv(chan, dmodel, 1, 1, 0, bias=True)
#         self.conv_out = Conv(dmodel, outchan, 1, 1, 0, bias=True)
#     def forward(self, x):
#         y = self.conv_in(x)
#         for block in self.blocks:
#             y = block(y)
#         y = self.conv_out(y)
#         y = torch.sigmoid(y.contiguous())
#         return y, 0
