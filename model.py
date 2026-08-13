"""T-AVFD inference model used by the accepted ICML checkpoint."""

from __future__ import annotations

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnablePrompt(nn.Module):
    """Frozen CLIP text encoder with the learned T-AVFD context tokens."""

    def __init__(self, clip_model: nn.Module, num_learnable_tokens: int = 5):
        super().__init__()
        self.pos_prompts = [
            "a real human face",
            "a bonafide face with expressive eyes",
            "a genuine face with natural mouth",
        ]
        self.neg_prompts = [
            "a fake human face",
            "a spoof face with dull eyes",
            "a forged face with unnatural mouth",
        ]
        self.num_learnable_tokens = int(num_learnable_tokens)
        self.clip_model = clip_model.float()
        for parameter in self.clip_model.parameters():
            parameter.requires_grad = False
        text_dim = self.clip_model.token_embedding.embedding_dim
        self.learnable_tokens = nn.Parameter(
            0.01 * torch.randn(1, self.num_learnable_tokens, text_dim)
        )

    def encode_prompts(self, prompts: list[str]) -> torch.Tensor:
        features = []
        device = self.learnable_tokens.device
        for prompt in prompts:
            tokens = clip.tokenize([prompt]).to(device)
            with torch.no_grad():
                base = self.clip_model.token_embedding(tokens).float()
            suffix = base[:, self.num_learnable_tokens :, :].clone()
            embedding = torch.cat((self.learnable_tokens, suffix), dim=1)
            positional = self.clip_model.positional_embedding[
                : embedding.size(1)
            ].float()
            encoded = (embedding + positional).permute(1, 0, 2)
            encoded = self.clip_model.transformer(encoded)
            encoded = self.clip_model.ln_final(encoded.permute(1, 0, 2))
            features.append(F.normalize(encoded[:, -1], dim=-1))
        return F.normalize(torch.stack(features).mean(dim=0), dim=-1)

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode_prompts(self.pos_prompts), self.encode_prompts(
            self.neg_prompts
        )


class MetaWeightGenerator(nn.Module):
    """Original ICML three-modality proposal gate."""

    def __init__(self):
        super().__init__()
        self.meta_net = nn.Sequential(
            nn.Linear(512 * 3, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Softmax(dim=-1),
        )

    def forward(
        self, visual: torch.Tensor, audio: torch.Tensor, semantic: torch.Tensor
    ) -> torch.Tensor:
        return self.meta_net(torch.cat((visual, audio, semantic), dim=-1))


class FusionModel(nn.Module):
    """Original T-AVFD fusion network with cached prompt inference."""

    def __init__(
        self,
        clip_model: nn.Module,
        feature_dim: int = 1024,
        global_dim: int = 768,
        tm_weights: list[float] | tuple[float, float, float] | None = None,
    ):
        super().__init__()
        self.local_proj = nn.Linear(768, 512)
        self.gl_proj = nn.Linear(global_dim, 512)
        self.audio_proj = nn.Linear(feature_dim, 512)
        self.visual_proj = nn.Linear(feature_dim, 512)
        self.kv_proj = nn.Linear(1024, 512)
        self.output_fc = nn.Sequential(
            nn.Linear(1536, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.text_feature = LearnablePrompt(clip_model)
        self.meta_weight_generator = MetaWeightGenerator()
        self.task_embedding = nn.Embedding(1, 3)
        initial = tm_weights if tm_weights is not None else [0.1, 0.1, -0.1]
        self.task_embedding.weight.data[0] = torch.tensor(initial)
        self._cached_prompts: tuple[torch.Tensor, torch.Tensor] | None = None

    def train(self, mode: bool = True):
        super().train(mode)
        self.text_feature.clip_model.eval()
        if mode:
            self._cached_prompts = None
        return self

    def prompt_features(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._cached_prompts is None:
            with torch.no_grad():
                self._cached_prompts = tuple(
                    value.detach() for value in self.text_feature()
                )
        return self._cached_prompts

    def forward(
        self,
        visual_features: torch.Tensor,
        audio_features: torch.Tensor,
        local_features: torch.Tensor | None = None,
        global_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del local_features  # Kept only for compatibility with the released NPZ format.
        if visual_features.dim() == 2:
            visual_features = visual_features.unsqueeze(0)
        if audio_features.dim() == 2:
            audio_features = audio_features.unsqueeze(0)
        if global_features is not None and global_features.dim() == 2:
            global_features = global_features.unsqueeze(0)
        if visual_features.size(1) != audio_features.size(1):
            raise ValueError("visual and audio must have the same sequence length")

        batch_size, length, _ = visual_features.shape
        text_positive, _ = self.prompt_features()
        audio = self.audio_proj(audio_features)
        visual = self.visual_proj(visual_features)
        text = self.local_proj(text_positive.float()).view(1, 1, -1)
        text = text.expand(batch_size, length, -1)

        if global_features is None:
            face = torch.zeros(
                batch_size,
                length,
                512,
                device=visual.device,
                dtype=visual.dtype,
            )
        else:
            face = self.gl_proj(global_features[:, :1]).expand(-1, length, -1)
        semantic = self.kv_proj(torch.cat((text, face), dim=-1))

        proposal = self.meta_weight_generator(visual, audio, semantic)
        task_prior = self.task_embedding.weight[0].view(1, 1, 3)
        weights = F.softmax(proposal + task_prior, dim=-1)
        fused = torch.cat(
            (
                weights[..., 0:1] * audio,
                weights[..., 1:2] * visual,
                weights[..., 2:3] * semantic,
            ),
            dim=-1,
        )
        return self.output_fc(fused)
