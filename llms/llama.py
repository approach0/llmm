from torch.nn import Module
from torch.nn.functional import silu, softmax

distributed = False
if distributed:
    from bmtrain import DistributedModule as DistributedModule
    from bmtrain import DistributedParameter as DistributedParameter
    from bmtrain import TransformerBlockList as ModuleList
    import sys
    sys.path.insert(0, './ModelCenter')
    from model_center.layer import Linear as DistributedLinear
    from model_center.layer import Embedding as DistributedEmbedding
else:
    from torch.nn import Module as DistributedModule
    from torch.nn import Parameter as DistributedParameter
    from torch.nn import ModuleList
    from torch.nn import Linear as DistributedLinear
    from torch.nn import Embedding as DistributedEmbedding

import math
import torch
from transformers import LlamaConfig


def _make_causal_mask(bsz, seq_len, past_seq_len, dtype, device):
    # make an infinity square-matrix mask
    mask = torch.full(
        (seq_len, seq_len),
        torch.tensor(torch.finfo(dtype).min, device=device),
    device=device)
    # fill lower diagonal by zeros
    mask_cond = torch.arange(mask.size(-1), device=device)
    inverted_mask_cond = (mask_cond + 1).view(mask.size(-1), 1)
    mask.masked_fill_(mask_cond < inverted_mask_cond, 0)
    mask = mask.to(dtype)
    # concatenate previous mask to a potentially rectangular mask
    if past_seq_len > 0:
        prev_mask = torch.zeros(seq_len, past_seq_len,
            dtype=dtype, device=device)
        mask = torch.cat([prev_mask, mask], dim=-1)
    # Expanding a tensor to a desired dim. This does not allocate
    # new memory, but only creates a new view on the existing tensor.
    return mask[None, None, :, :].expand(
        bsz, 1, seq_len, seq_len + past_seq_len)


def _expand_mask(mask, dtype, seq_len):
    bsz, src_len = mask.size()
    expanded_mask = mask[:, None, None, :].expand(
        bsz, 1, seq_len, src_len).to(dtype)
    inverted_mask = 1.0 - expanded_mask
    return inverted_mask.masked_fill(
        inverted_mask.to(torch.bool), torch.finfo(dtype).min)


class LlamaRMSNorm(DistributedModule):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = DistributedParameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, states):
        input_dtype = states.dtype
        variance = states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        rsqrt = torch.rsqrt(variance + self.variance_epsilon)
        states = states * rsqrt
        return (self.weight * states).to(input_dtype)


class LlamaRotaryEmbedding(DistributedModule):
    def __init__(self, dim,
        max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        theta = 1.0 / (base ** (
            torch.arange(0, dim, 2).float().to(device) / dim)
        )
        # theta: [head_H // 2]

        # Buffer will show up in state_dict(), no need to save it here.
        # self.register_buffer("theta", theta, persistent=False)

        t = torch.arange(max_position_embeddings,
            device=theta.device, dtype=theta.dtype)
        # t: [max_seq_len]

        cache = torch.einsum("i,j->ij", t, theta)
        # cache_{i,j} = t_i * theta_j
        # cache: [max_seq_len, head_H // 2]

        cache = torch.cat((cache, cache), dim=-1)
        # cache: [max_seq_len, head_H]

        dtype = torch.get_default_dtype()
        self.cos_cached = cache.cos().to(dtype)
        self.sin_cached = cache.sin().to(dtype)
        # {sin, cos}_cached: [max_seq_len, head_H]

    def forward(self, tot_seq_len):
        # get partial cache that matches the input tot_seq_len
        return (
            self.cos_cached[:tot_seq_len, ...],
            self.sin_cached[:tot_seq_len, ...]
        )

    @staticmethod
    def apply(q, k, cos, sin, position_ids, timestep=0):
        cos = cos.to(device=q.device, dtype=q.dtype)
        sin = sin.to(device=q.device, dtype=q.dtype)
        # {sin, cos}: [tot_seq_len, head_H]

        # position_ids: [bs, seq_len]
        sin = sin[position_ids].unsqueeze(1)
        cos = cos[position_ids].unsqueeze(1)
        # Because position_ids are either full initial ids
        # or the current decoding index, the {sin, cos} now
        # becomes of partial tot_seq_len, i.e., seq_len:
        # {sin, cos}: [bs, 1, seq_len, head_H]
        # Now, {sin, cos} @ [bs, 1, m, k] is:
        #   cos(m * theta_k) when k < head_H//2
        #   cos(m * theta_{k - head_H//2}) when otherwise

        # split the head_H dimension, so that
        # x @ [bs, heads, m, :] is: [-x_m[mid:]; x_m[:mid]]
        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        # Now, apply RoPE using element-wise product...
        # Different original author blog post, HuggingFace Llama
        # implementation apply rotation on the 2-D slice
        #   [x_{d}, x_{head_H//2 + d}]
        # instead of
        #   [x_{2d+0}, x_{2d+1}]

        # given a 4-D vector x for example (x can be q or k):
        #
        # | cos(m θ_0) -sin(m θ_0)                      |   |x0|
        # | sin(m θ_0)  cos(m θ_0)                      | * |x2|
        # |                      cos(m θ_1) -sin(m θ_1) |   |x1|
        # |                      sin(m θ_1)  cos(m θ_1) |   |x3|
        #
        # which equals
        #
        # | x0 cos(m θ_0) - x2 sin(m θ_0) |
        # | x2 cos(m θ_0) + x0 sin(m θ_0) |
        # | x1 cos(m θ_1) - x3 sin(m θ_1) |
        # | x3 cos(m θ_1) + x1 sin(m θ_1) |
        #
        # and if we reorder it back, the transformed vector is:
        #
        # | x0 cos(m θ_0) - x2 sin(m θ_0) |
        # | x1 cos(m θ_1) - x3 sin(m θ_1) |
        # | x2 cos(m θ_0) + x0 sin(m θ_0) |
        # | x3 cos(m θ_1) + x1 sin(m θ_1) |
        #
        # which is essentially what is written below:
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)

        return q_embed, k_embed


class SiLUActivation(DistributedModule):
    def forward(self, input):
        return silu(input)


class LlamaMLP(DistributedModule):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = DistributedLinear(hidden_size, intermediate_size, bias=False)
        self.down_proj = DistributedLinear(intermediate_size, hidden_size, bias=False)
        self.up_proj = DistributedLinear(hidden_size, intermediate_size, bias=False)
        self.act_fn = SiLUActivation()

    def forward(self, x):
        # SwiGLU: https://arxiv.org/abs/2002.05202v1
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class LlamaAttention(DistributedModule):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size # H
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads # head_H
        H = self.num_heads * self.head_dim
        self.q_proj = DistributedLinear(self.hidden_size, H, bias=False)
        self.k_proj = DistributedLinear(self.hidden_size, H, bias=False)
        self.v_proj = DistributedLinear(self.hidden_size, H, bias=False)
        self.o_proj = DistributedLinear(H, self.hidden_size, bias=False)
        self.rotary_emb = LlamaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings
        )

    def forward(self, hidden_states,
        attention_mask=None,
        position_ids=None,
        past_cache=None,
        use_cache=False,
        timestep=0):
        bsz, seq_len, _ = hidden_states.size()
        # split inputs into heads of dimension [B, heads, seq_len, head_H]
        split_dim = (bsz, seq_len, self.num_heads, self.head_dim)
        Q = self.q_proj(hidden_states).view(*split_dim).transpose(1, 2)
        K = self.k_proj(hidden_states).view(*split_dim).transpose(1, 2)
        V = self.v_proj(hidden_states).view(*split_dim).transpose(1, 2)

        tot_seq_len = seq_len
        if past_cache is not None:
            tot_seq_len += past_cache[0].shape[-2]

        # get rotary position embedding
        cos, sin = self.rotary_emb(tot_seq_len)
        # cos, sin: [1, 1, tot_seq_len, head_H]
        Q, K = LlamaRotaryEmbedding.apply(
            Q, K, cos, sin, position_ids, timestep=timestep)
        # Q, K: [B, heads, tot_seq_len, head_H]

        if past_cache is not None:
            # reuse past K, V, self_attention
            past_key, past_val = past_cache
            # past_key or past_val: [B, heads, past_seq_len, head_H]
            K = torch.cat([past_key, K], dim=2)
            V = torch.cat([past_val, V], dim=2)
            # new K or V: [B, heads, tot_seq_len, head_H]
        # set either the initial or concatenated past_cache
        past_cache = (K, V) if use_cache else None

        # apply scaled dot-product self-attention
        attn_W = torch.matmul(
            Q, K.transpose(2, 3)
        ) / math.sqrt(self.head_dim)
        assert attn_W.size() == (bsz, self.num_heads, seq_len, tot_seq_len)

        # apply attention_mask!
        if attention_mask is not None:
            assert attention_mask.size() == (bsz, 1, seq_len, tot_seq_len)
            attn_W = attn_W + attention_mask

        # upcast to fp32 before softmax and downcast back to the original dtype
        attn_W = softmax(attn_W, dim=-1, dtype=torch.float32).to(Q.dtype)

        # apply attention weights to Value
        attn_out = torch.matmul(attn_W, V)
        assert attn_out.size() == (bsz, self.num_heads, seq_len, self.head_dim)

        # join heads
        attn_out = attn_out.transpose(1, 2)
        attn_out = attn_out.reshape(bsz, seq_len, self.hidden_size)

        # attention output projection
        attn_out = self.o_proj(attn_out)
        return attn_out, past_cache


class LlamaDecoderLayer(Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config)
        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
        )
        self.norm1 = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm2 = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states,
        attention_mask=None,
        position_ids=None,
        past_cache=None,
        use_cache=False,
        timestep=0):

        # Background: https://arxiv.org/pdf/2002.04745.pdf
        #
        # Original Post-LayerNorm         Pre-LayerNorm Layer
        #
        #        x(l+1)                       x(l+1)
        #         |                            |
        #      LayerNorm                      (+)----*
        #         |                            |     |
        #        (+)----*                     FFN    |
        #         |     |                      |     |
        #        FFN    |                LayerNorm   |
        #         |     |                      |     |
        #         *-----*                      *-----*
        #         |                            |
        #      LayerNorm                      (+)----*
        #         |                            |     |
        #        (+)----*                Attention   |
        #         |     |                      |     |
        #     Attention |                LayerNorm   |
        #         |     |                      |     |
        #         *-----*                      *-----*
        #         |                            |
        #        x(l)                         x(l)

        residual = hidden_states
        hidden_states = self.norm1(hidden_states)

        hidden_states, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_cache=past_cache,
            use_cache=use_cache,
            timestep=timestep
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        outputs = (hidden_states,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs


class LlamaModel(Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = DistributedEmbedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        if distributed:
            from bmtrain import CheckpointBlock
            self.layers = ModuleList([
                CheckpointBlock(LlamaDecoderLayer(config))
                for _ in range(config.num_hidden_layers)
            ])
        else:
            self.layers = ModuleList([
                LlamaDecoderLayer(config)
                for _ in range(config.num_hidden_layers)
            ])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def _prepare_decoder_attention_mask(self,
        attention_mask, static_embeds, past_seq_len):
        bsz, seq_len, _ = static_embeds.shape
        # make a rectangular causal mask with past history
        combined_attention_mask = _make_causal_mask(
            bsz, seq_len, past_seq_len,
            static_embeds.dtype,
            static_embeds.device,
        )

        if attention_mask is not None:
            # add (expanded) attention_mask to causal mask
            expanded_mask = _expand_mask(
                attention_mask,
                static_embeds.dtype,
                seq_len
            ).to(static_embeds.device)
            combined_attention_mask += expanded_mask

        return combined_attention_mask

    def forward(self, input_ids,
        attention_mask=None,
        position_ids=None,
        past_caches=None,
        use_cache=None,
        timestep=0):
        # calculate various lengths
        batch_size, seq_length = input_ids.shape
        tot_seq_len = seq_length
        past_seq_len = 0
        if past_caches is not None:
            # past_caches[layer][k/v]:
            # [B, heads, past_seq_len, head_H]
            past_seq_len = past_caches[0][0].shape[2]
            tot_seq_len += past_seq_len

        if position_ids is None:
            position_ids = torch.arange(
                past_seq_len, tot_seq_len,
                dtype=torch.long,
                device=input_ids.device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()
        # position_ids: [B, seq_len] using this timestep as the start pos

        # [B, seq_len, H] non-contextual word embeddings
        static_embeds = self.embed_tokens(input_ids)

        assert attention_mask.shape == (batch_size, tot_seq_len)
        # convert linear attention mask to causal (rectangular) attention
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, # [B, tot_seq_len]
            static_embeds, # [B, seq_len, H]
            past_seq_len
        )
        assert attention_mask.shape == (batch_size, 1,
            seq_length, tot_seq_len)
        # Example causal attention_mask:
        #
        # Case 1 (when timestep = 0, tot_seq_len = seq_len):
        # |  0  -inf ... -inf -inf -inf | (q_{t=0})
        # |  0    0  ... -inf -inf -inf |
        # ...
        # |  0    0  ...   0    0  -inf |
        # |  0    0  ...   0    0    0  | (q_{t=seq_len})
        #
        # Case 2 (when timestep > 0, tot_seq_len = past_seq_len + 1):
        # |  0    0  ...   0    0    0   0 | (all zeros)

        # decoder layers
        hidden_states = static_embeds # [B, seq_len, H]
        new_caches = () if use_cache else None
        for idx, decoder_layer in enumerate(self.layers):
            # get "past_cache" at this idx-th layer
            past_cache = (past_caches[idx]
                if past_caches is not None else None)

            # layer_outputs = (hidden_states, layer_cache)
            #   hidden_states: [B, seq_len, H]
            #   layer_cache = (K, V)
            #     K: [B, heads, tot_seq_len, head_H]
            #     V: [B, heads, tot_seq_len, head_H]
            layer_outputs = decoder_layer(
                hidden_states,                 # [B, seq_len, H]
                attention_mask=attention_mask, # [B, 1, seq_len, tot_seq_len]
                position_ids=position_ids,     # [B, seq_len]
                past_cache=past_cache,     # [B, heads, past_seq_len, head_H]
                use_cache=use_cache,
                timestep=timestep
            )
            hidden_states = layer_outputs[0] # next recurrent states
            if use_cache:
                layer_cache = layer_outputs[1]
                new_caches += (layer_cache,)

        hidden_states = self.norm(hidden_states)
        return hidden_states, new_caches


class LlamaForCausalLM(Module):
    def __init__(self, config):
        super().__init__()
        self.model = LlamaModel(config)
        self.lm_head = DistributedLinear(
            config.hidden_size, config.vocab_size, bias=False)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_caches=None,
        labels=None,
        use_cache=None,
        timestep=0):
        # invoke Llama model
        hidden_states, new_caches = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_caches=past_caches,
            use_cache=use_cache,
            timestep=timestep
        )
        # convert to sparse logits
        logits = self.lm_head(hidden_states) # [B, seq_len, vocab]
        return logits, new_caches
