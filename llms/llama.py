import math
import torch
from torch import nn
from transformers import LlamaConfig


def _make_causal_mask(input_ids_shape, dtype, device, past_key_values_length):
    bsz, tgt_len = input_ids_shape
    # make an infinity square-matrix mask
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
    # fill lower diagnoal by zeros
    mask_cond = torch.arange(mask.size(-1), device=device)
    inverted_mask_cond = (mask_cond + 1).view(mask.size(-1), 1)
    mask.masked_fill_(mask_cond < inverted_mask_cond, 0)
    mask = mask.to(dtype)
    # concate previous mask to a potentially rect mask
    if past_key_values_length > 0:
        prev_mask = torch.zeros(tgt_len, past_key_values_length,
            dtype=dtype, device=device)
        mask = torch.cat([prev_mask, mask], dim=-1)
    # Expanding a tensor does not allocate new memory,
    # but only creates a new view on the existing tensor.
    return mask[None, None, :, :].expand(
        bsz, 1, tgt_len, tgt_len + past_key_values_length)


def _expand_mask(mask, dtype, tgt_len):
    bsz, src_len = mask.size()
    expanded_mask = mask[:, None, None, :].expand(
        bsz, 1, tgt_len, src_len).to(dtype)
    inverted_mask = 1.0 - expanded_mask
    return inverted_mask.masked_fill(
        inverted_mask.to(torch.bool), torch.finfo(dtype).min)


def _apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        rsqrt = torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = hidden_states * rsqrt
        return (self.weight * hidden_states).to(input_dtype)


class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len_cached = None
        self.register_sincos_buf(max_position_embeddings)

    def register_sincos_buf(self, max_position_embeddings=None):
        if max_position_embeddings is not None:
            self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached,
            device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        dtype = torch.get_default_dtype()
        cos = emb.cos()[None, None, :, :].to(dtype)
        sin = emb.sin()[None, None, :, :].to(dtype)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def forward(self, x, seq_len=None):
        if self.cos_cached.is_meta:
            self.register_sincos_buf()
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


class SiLUActivation(nn.Module):
    def forward(self, input):
        return nn.functional.silu(input)


class LlamaMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.act_fn = SiLUActivation()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class LlamaAttention(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)

    def forward(self, hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        use_cache=False):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = _apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids)
        # [bsz, nh, t, hd]

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        assert attn_weights.size() == (bsz, self.num_heads, q_len, kv_seq_len)
        if attention_mask is not None:
            assert attention_mask.size() == (bsz, 1, q_len, kv_seq_len)
            attn_weights = attn_weights + attention_mask
            dtype_min = torch.tensor(
                torch.finfo(attn_weights.dtype).min, device=attn_weights.device, dtype=attn_weights.dtype
            )
            attn_weights = torch.max(attn_weights, dtype_min)

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        assert attn_output.size() == (bsz, self.num_heads, q_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, past_key_value


class LlamaDecoderLayer(nn.Module):
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
        past_key_value=None,
        use_cache=False):

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
            past_key_value=past_key_value,
            use_cache=use_cache,
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


class LlamaModel(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def _prepare_decoder_attention_mask(self,
        attention_mask,
        input_shape,
        inputs_embeds,
        past_key_values_length):
        bsz, seq_len = input_shape
        combined_attention_mask = _make_causal_mask(
            input_shape,
            inputs_embeds.dtype,
            device=inputs_embeds.device,
            past_key_values_length=past_key_values_length,
        )

        if attention_mask is not None:
            # apply given attention_mask to causual mask
            expanded_mask = _expand_mask(
                attention_mask,
                inputs_embeds.dtype,
                seq_len
            ).to(inputs_embeds.device)
            combined_attention_mask += expanded_mask

        return combined_attention_mask

    def forward(self, input_ids,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=None):

        batch_size, seq_length = input_ids.shape
        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past += past_key_values_length

        if position_ids is None:
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=input_ids.device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        inputs_embeds = self.embed_tokens(input_ids)

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, # [B, seq_len]
            (batch_size, seq_length),
            inputs_embeds, # [B, seq_len, 4096]
            past_key_values_length
        )
        # attention_mask (e.g., seq_length=3):
        # | 0 inf inf |
        # | 0  0  inf |
        # | 0  0   0  |
        assert attention_mask.shape == (batch_size, 1, seq_length, seq_length)

        # decoder layers
        hidden_states = inputs_embeds
        next_cache = () if use_cache else None
        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = past_key_values[idx] if past_key_values is not None else None

            # layer_outputs = (hidden_states, decoder_cache)
            # hidden_states: [B, seq_len, 4096]
            # decoder_cache = (key_states, value_states)
            # key_states: [B, n_heads, seq_len, hidden_dim]
            # value_states: [B, n_heads, seq_len, hidden_dim]
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0] # for recurrent inputs
            decoder_cache = layer_outputs[1] # for saving and speedup
            if use_cache:
                next_cache += (decoder_cache,)

        hidden_states = self.norm(hidden_states)
        next_cache = next_cache if use_cache else None
        return hidden_states, next_cache


class LlamaForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        labels=None,
        use_cache=None):

        hidden_states, next_cache = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache
        )

        logits = self.lm_head(hidden_states)
        return logits, next_cache
