import torch.nn as nn


class Adapter(nn.Module):
    def __init__(
            self, ds_factor, hidden_dim, ln_after=False, ln_before=False, dropout=0.1
    ):
        super().__init__()
        assert not hidden_dim % ds_factor
        self.down = nn.Linear(hidden_dim, hidden_dim // ds_factor)
        self.act = nn.ReLU()
        self.up = nn.Linear(hidden_dim // ds_factor, hidden_dim)
        self.apply(self.init_weights)
        self.ln_after = ln_after
        self.ln_before = ln_before
        self.dropout = dropout
        if ln_after or ln_before:
            self.ln = nn.LayerNorm(hidden_dim)
        if dropout:
            self.dropout = nn.Dropout(dropout)

    def init_weights(self, module, std=0.02):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(self, hidden_states):
        if self.ln_before:
            residual = self.ln(hidden_states)
            residual = self.down(residual)
        else:
            residual = self.down(hidden_states)
        residual = self.act(residual)
        if self.dropout:
            residual = self.dropout(residual)
        residual = self.up(residual)
        if self.ln_after:
            residual = self.ln(hidden_states)
        return hidden_states + residual
