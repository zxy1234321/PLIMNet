import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights
from transformers import AutoConfig, AutoModel

from .config import ModelConfig, TASKS, LINEAGES, LINEAGE_TASKS


class ImageBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        network = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.body = nn.Sequential(*list(network.children())[:-1])
        self.feature_dim = network.fc.in_features

    def forward(self, images):
        return self.body(images).flatten(1)


class PLIMNet(nn.Module):
    def __init__(self, config=None, pretrained_images=True, text_config=None, local_files_only=False):
        super().__init__()
        self.config = config or ModelConfig()
        cfg = self.config
        self.task_names = TASKS
        self.task_to_idx = {name: index for index, name in enumerate(TASKS)}
        self.backbone = ImageBackbone(pretrained_images)
        if text_config is None:
            self.text_encoder = AutoModel.from_pretrained(cfg.text_model, local_files_only=local_files_only)
        else:
            values = dict(text_config)
            model_type = values.pop("model_type")
            self.text_encoder = AutoModel.from_config(AutoConfig.for_model(model_type, **values))
        for parameter in self.text_encoder.parameters():
            parameter.requires_grad = False
        layers = getattr(getattr(self.text_encoder, "encoder", None), "layer", None)
        if layers is None or cfg.text_unfreeze_layers > len(layers):
            raise ValueError("The text encoder must expose BERT encoder layers")
        if cfg.text_unfreeze_layers:
            for layer in layers[-cfg.text_unfreeze_layers:]:
                for parameter in layer.parameters():
                    parameter.requires_grad = True
        hidden = cfg.hidden_dim
        self.single_image_proj = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, hidden),
                                               nn.GELU(), nn.Dropout(cfg.dropout))
        text_hidden = self.text_encoder.config.hidden_size
        self.text_proj = nn.Sequential(nn.LayerNorm(text_hidden), nn.Linear(text_hidden, hidden),
                                       nn.GELU(), nn.Dropout(cfg.dropout))
        self.image_query_attn = nn.MultiheadAttention(hidden, cfg.attention_heads,
                                                      dropout=cfg.attention_dropout, batch_first=True)
        self.text_query_attn = nn.MultiheadAttention(hidden, cfg.attention_heads,
                                                     dropout=cfg.attention_dropout, batch_first=True)
        self.shared_head = nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(cfg.dropout),
                                         nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(cfg.dropout))
        self.global_task_head = nn.Linear(hidden, len(TASKS))
        self.lineage_head = nn.Linear(hidden, len(LINEAGES))
        branch_input = hidden + len(LINEAGES)
        self.mixed_task_heads = nn.ModuleDict({"lh": nn.Sequential(
            nn.LayerNorm(branch_input), nn.Dropout(cfg.dropout), nn.Linear(branch_input, hidden),
            nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(hidden, 2))})
        self.branch_indices = {name: [self.task_to_idx[task] for task in LINEAGE_TASKS[name]]
                               for name in LINEAGES}
        self.branch_mlps = nn.ModuleDict({name: nn.Sequential(
            nn.LayerNorm(branch_input), nn.Dropout(cfg.dropout), nn.Linear(branch_input, hidden),
            nn.GELU(), nn.Dropout(cfg.dropout)) for name in LINEAGES})
        self.branch_heads = nn.ModuleDict({name: nn.Linear(hidden, len(self.branch_indices[name]))
                                          for name in LINEAGES})

    @staticmethod
    def masked_mean(tokens, mask):
        weights = mask.unsqueeze(-1).float()
        return (tokens * weights).sum(1) / weights.sum(1).clamp_min(1.0)

    def encode_fused(self, images, text_inputs):
        if images.ndim != 5 or images.shape[1:3] != (5, 3):
            raise ValueError("Expected images with shape (batch, 5, 3, height, width)")
        batch, planes, channels, height, width = images.shape
        features = self.backbone(images.reshape(batch * planes, channels, height, width))
        image_tokens = self.single_image_proj(features.reshape(batch, planes, 512))
        text_tokens = self.text_proj(self.text_encoder(**text_inputs).last_hidden_state)
        mask = text_inputs["attention_mask"]
        image_context, _ = self.image_query_attn(image_tokens, text_tokens, text_tokens,
                                                  key_padding_mask=~mask.bool())
        text_context, _ = self.text_query_attn(text_tokens, image_tokens, image_tokens)
        return image_context.mean(1) + self.masked_mean(text_context, mask)

    def forward(self, images, text_inputs):
        shared = self.shared_head(self.encode_fused(images, text_inputs))
        task_logits = self.global_task_head(shared).clone()
        lineage_logits = self.lineage_head(shared)
        gates = lineage_logits.sigmoid().detach()
        branch_input = torch.cat([shared, gates], dim=1)
        for index, name in enumerate(LINEAGES):
            residual = self.branch_heads[name](self.branch_mlps[name](branch_input))
            task_logits[:, self.branch_indices[name]] += gates[:, index:index + 1] * residual
        weights = gates[:, [0, 2]]
        weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-6)
        task_logits[:, self.task_to_idx["lh"]] += (weights * self.mixed_task_heads["lh"](branch_input)).sum(1)
        return lineage_logits, task_logits
