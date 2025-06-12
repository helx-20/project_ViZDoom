import torch.nn as nn
from bert import Block, Config

def define_Transformer(input_dim, output_dim, m_tokens, h_dim=64):
    """
    Define a Transformer network for ViZDoom.
    Args:
        input_dim (int): Dimension of the input features.
        output_dim (int): Dimension of the output features.
        m_tokens (int): Number of tokens in the sequence.
    Returns:
        nn.Sequential: A sequential model containing the transformer backbone and output prediction layer.
    """

    # define backbone networks
    bert_cfg = Config()
    bert_cfg.dim = h_dim
    bert_cfg.n_layers = 2
    bert_cfg.n_heads = 2
    bert_cfg.max_len = m_tokens
    Backbone = Transformer(input_dim=input_dim, m_tokens=m_tokens, cfg=bert_cfg)
    # define output prediction network
    P = nn.Linear(h_dim*m_tokens, output_dim, bias=True)

    return nn.Sequential(Backbone, P)

class Transformer(nn.Module):
    """ Transformer with Self-Attentive Blocks"""

    def __init__(self, input_dim, m_tokens, cfg=Config()):
        super().__init__()
        self.in_net = nn.Linear(input_dim, cfg.dim, bias=True)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.m_tokens = m_tokens

    def forward(self, x):
        # shape x: batch_size x m_token x m_state
        h = self.in_net(x)
        for block in self.blocks:
            h = block(h, None)
        h = h.view(h.shape[0], -1)
        return h
