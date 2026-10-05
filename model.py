import torch
import torch.nn as nn
from transformers import AutoModel

class MERTFineTune(nn.Module):
    def __init__(self, model_name: str, n_classes: int = 6):
        super(MERTFineTune, self).__init__()

        self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        # Not freeze
        self.encoder.requires_grad_(True)
        # Freeze
        # self.encoder.requires_grad_(False)
        
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256), #256
            nn.ReLU(),
            nn.Dropout(0.3), # 0.3
            nn.Linear(256, n_classes),
        )

    def forward(self, input_values, batch_size, num_chunks, attention_mask) -> torch.Tensor:
        outputs = self.encoder(input_values=input_values, attention_mask=attention_mask, return_dict=True)

        chunk_features = outputs.last_hidden_state.mean(dim=1)
        chunk_features = chunk_features.reshape(batch_size, num_chunks, -1)
        recording_features = chunk_features.mean(dim=1)

        return self.classifier(recording_features)