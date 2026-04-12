import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class SparsemaxFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, dim):
        dim = dim % input.dim()  # normalize negative dim

        # sort descending along dim
        z_sorted, _ = torch.sort(input, descending=True, dim=dim)

        # cumulative sums
        # C_j = sum of top-j elements
        cumsum = torch.cumsum(z_sorted, dim=dim)

        # rho = 1, 2, ..., d broadcast to the target dim
        d = input.shape[dim]
        shape = [1] * input.dim()
        shape[dim] = d
        rho = torch.arange(1, d + 1, dtype=input.dtype, device=input.device).view(shape)

        # support conditio
        # 1 + rho * z_(rho) > C_(rho)
        # holds for a prefix of sorted elements
        support_mask = (1.0 + rho * z_sorted > cumsum)

        # k = number of elements in the support
        k = support_mask.sum(dim=dim, keepdim=True).clamp(min=1)

        # tau = (C_k - 1) / k 
        # gather cumsum at index k-1 (0-indexed)
        cumsum_at_k = torch.gather(cumsum, dim, k - 1)
        tau = (cumsum_at_k - 1.0) / k.to(input.dtype)

        output = torch.clamp(input - tau, min=0.0)

        ctx.save_for_backward(output)
        ctx.dim = dim
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        dim = ctx.dim

        support = (output > 0).to(grad_output.dtype)
        support_size = support.sum(dim=dim, keepdim=True).clamp(min=1)
        # mean gradient over the support, then subtract from each support element
        v_hat = (grad_output * support).sum(dim=dim, keepdim=True) / support_size
        grad_input = support * (grad_output - v_hat)

        return grad_input, None  # none for dim (not a tensor)


def sparsemax(input, dim=-1):
    return SparsemaxFunction.apply(input, dim)


class BaseAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.LLMC_RESIDUAL_SCALE_FLAG = 1

        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # not really a 'bias', more of a mask, but following the OpenAI/HF naming though
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))


class SoftmaxAttention(BaseAttention):

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q, k, v = q.float(), k.float(), v.float()

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v

        y = y.to(x.dtype)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y
    

class SparsemaxAttention(BaseAttention):

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q, k, v = q.float(), k.float(), v.float()

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = sparsemax(att, dim=-1) # implement sparsemax forward and backward
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) to (B, nh, T, hs)

        y = y.to(x.dtype)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

    
class DynamicReluAttention(BaseAttention):

    def __init__(self, config):
        super().__init__(config)
        self.beta = nn.Parameter(torch.randn(config.n_head) / 10.0)

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q, k, v = q.float(), k.float(), v.float()

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, -10000.0)

        # detach amax
        # it's a reference point, not part of the learned transformation
        # gradients through amax zero out the max element and corrupt other positions
        # tau = F.sigmoid(self.beta.float()).view(1, self.n_head, 1, 1)
        # att = att - att.amax(dim=-1, keepdim=True).detach() + 1.0
        # relu_att = F.relu(att - tau)
        # att = relu_att / relu_att.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        sig = torch.sigmoid(self.beta.float()).view(1, self.n_head, 1, 1)
        row_max = att.max(dim=-1, keepdim=True).values  # shape (B, H, T, 1)
        
        relu_scores = F.relu(att - row_max + sig)
        denom = relu_scores.sum(dim=-1, keepdim=True)  # (B, H, T, 1)
        att = relu_scores / denom.clamp(min=1e-7)

        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = y.to(x.dtype)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y