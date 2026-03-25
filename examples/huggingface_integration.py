"""Example: Golden Pendulum MTL with Hugging Face Transformers.

Shows how to integrate golden-ratio gradient balancing into a
custom HuggingFace Trainer for multi-task fine-tuning.

Requires: pip install transformers datasets
"""

import torch
import torch.nn as nn

from golden_pendulum import GoldenPendulumMTL


class MultiTaskBertModel(nn.Module):
    """Example multi-task model built on a HuggingFace backbone.

    Replace with your actual model — this is a template.
    """

    def __init__(self, backbone_dim: int = 768, n_classes: int = 3):
        super().__init__()
        # In practice: self.backbone = AutoModel.from_pretrained("bert-base-uncased")
        self.backbone = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=backbone_dim, nhead=8, batch_first=True),
            num_layers=2,
        )
        self.embed = nn.Linear(32, backbone_dim)
        self.head_cls = nn.Linear(backbone_dim, n_classes)  # Classification
        self.head_ner = nn.Linear(backbone_dim, 9)  # NER tags
        self.head_sim = nn.Linear(backbone_dim, 128)  # Sentence similarity

    def forward(self, x):
        h = self.backbone(self.embed(x))
        h_pool = h.mean(dim=1)
        return {
            "cls_logits": self.head_cls(h_pool),
            "ner_logits": self.head_ner(h),  # Per-token
            "sim_embeddings": self.head_sim(h_pool),
        }


def train_with_golden_pendulum():
    """Training loop with Golden Pendulum gradient balancing."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MultiTaskBertModel(backbone_dim=256).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    # Initialize Golden Pendulum for 3 tasks
    balancer = GoldenPendulumMTL(n_tasks=3, lam=0.5)

    print("Golden-ratio targets for K=3:")
    # For K=3: (0.186, 0.302, 0.488) — no pair has a rational ratio
    targets = [f"{w:.3f}" for w in [0.186, 0.302, 0.488]]
    print(f"  classification={targets[0]}, ner={targets[1]}, similarity={targets[2]}")
    print()

    for step in range(100):
        optimizer.zero_grad()

        # Synthetic batch (replace with your DataLoader)
        batch_size, seq_len = 16, 64
        x = torch.randn(batch_size, seq_len, 32, device=device)
        cls_labels = torch.randint(0, 3, (batch_size,), device=device)
        ner_labels = torch.randint(0, 9, (batch_size, seq_len), device=device)

        outputs = model(x)

        # Compute per-task losses
        losses = {
            "classification": nn.functional.cross_entropy(outputs["cls_logits"], cls_labels),
            "ner": nn.functional.cross_entropy(
                outputs["ner_logits"].reshape(-1, 9), ner_labels.reshape(-1)
            ),
            "similarity": info_nce(outputs["sim_embeddings"]),
        }

        # Golden Pendulum backward (replaces loss.backward())
        weights = balancer.backward(losses, model)
        optimizer.step()

        if step % 25 == 0:
            print(f"Step {step:3d} | balance={balancer.weight_balance_ratio:.3f} | {weights}")

    print(f"\nFinal mean weights: {balancer.mean_weights()}")


def info_nce(embeddings, temperature=0.1):
    embeddings = nn.functional.normalize(embeddings, dim=1)
    sim = embeddings @ embeddings.T / temperature
    labels = torch.arange(len(embeddings), device=embeddings.device)
    return nn.functional.cross_entropy(sim, labels)


if __name__ == "__main__":
    train_with_golden_pendulum()
