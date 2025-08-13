import torch
import torch.nn as nn

class DecisionTransformer(nn.Module):
    def __init__(self,
                 state_dim=65,
                 action_dim=1,
                 max_context_length=None,
                 max_ep_length=336,                 
                 model_dim=128,
                 device='cpu'
                 ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_context_length = max_context_length
        self.max_ep_length = max_ep_length
        self.model_dim = model_dim
        self.device = device


        self.embed_timestep = nn.Embedding(self.max_ep_length, self.model_dim)
        self.embed_return = torch.nn.Linear(1, self.model_dim)
        self.embed_state = torch.nn.Linear(self.state_dim, self.model_dim)
        self.embed_action = torch.nn.Linear(self.action_dim, self.model_dim)
        self.embed_ln = nn.LayerNorm(self.model_dim)

        self.predict_action = nn.Sequential(
            nn.Linear(self.model_dim, self.action_dim),
            nn.Tanh()
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.model_dim,
            nhead=8,
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=6)

    def forward(self, states, actions, returns_to_go, timesteps, padding_mask=None):
        batch_size, seq_length = states.shape[0], states.shape[1]

        if padding_mask is None:
            padding_mask = torch.ones((batch_size, seq_length), dtype=torch.float32)
        
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.model_dim)
        stacked_inputs = self.embed_ln(stacked_inputs)
        dummy_memory = torch.zeros(size=(1, seq_length, self.model_dim), device=self.device)

        causal_mask = torch.triu(torch.full((3*seq_length, 3*seq_length), float('-inf'), device=self.device), diagonal=1)

        stacked_padding_mask = torch.stack((padding_mask,padding_mask,padding_mask), dim=1).permute(0,2,1).reshape(batch_size,3*seq_length)

        x = self.transformer(tgt=stacked_inputs,
                             memory=dummy_memory, 
                             tgt_mask=causal_mask,
                             tgt_key_padding_mask=stacked_padding_mask)
        x = x.reshape(batch_size, seq_length, 3, self.model_dim).permute(0, 2, 1, 3)

        return self.predict_action(x[:,1])

    def get_action(self, states, actions, rtg, timesteps):

        # Add batch dimension and reshape to [1, seq_len, state_dim] so input matches Transformer input format
        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.action_dim)
        rtg = rtg.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.max_context_length is not None:
            states = states[:,-self.max_context_length:]
            actions = actions[:,-self.max_context_length:]
            rtg = rtg[:,-self.max_context_length:]
            timesteps = timesteps[:,-self.max_context_length:]

            # pad all tokens to sequence length
            padding_mask = torch.cat([torch.zeros(self.max_context_length-states.shape[1], device=self.device, dtype=torch.float32), 
                                      torch.ones(states.shape[1],dtype=torch.long, device=self.device)]).reshape(1,-1)
            states = torch.cat([torch.zeros((states.shape[0], self.max_context_length-states.shape[1], self.state_dim), device=self.device), 
                                states], dim=1)
            actions = torch.cat([torch.zeros((actions.shape[0], self.max_context_length - actions.shape[1], self.action_dim), device=self.device),
                                 actions], dim=1)
            rtg = torch.cat([torch.zeros((rtg.shape[0], self.max_context_length-rtg.shape[1], 1), device=self.device), 
                             rtg], dim=1)
            timesteps = torch.cat([torch.zeros((timesteps.shape[0], self.max_context_length-timesteps.shape[1]), dtype=torch.long, device=self.device),
                                   timesteps], dim=1)
        else:
            padding_mask = None

        action_preds = self.forward(states, actions, rtg, timesteps, padding_mask)
        
        return action_preds[0,-1]