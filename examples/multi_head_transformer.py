"""Example: Golden Pendulum MTL with a multi-head transformer.

Shows how to integrate with a realistic multi-head architecture where
task losses have disparate magnitudes (the exact failure mode that
Golden Pendulum solves).
"""

import torch
import torch.nn as nn

from golden_pendulum import GoldenPendulumMTL, WeightLogger


class SharedTransformerBackbone(nn.Module):
    """Minimal transformer backbone with shared representations."""

    def __init__(self, d_model: int = 128, n_heads: int = 4, n_layers: int = 2, seq_len: int = 32):
        super().__init__()
        self.embed = nn.Linear(16, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x):
        return self.encoder(self.embed(x))


class MultiHeadModel(nn.Module):
    """4-head model mimicking the paper's Phase A setup.

    Heads have intentionally different output scales:
    - Regression: MSE loss ~0.01
    - Classification: BCE loss ~0.69
    - Ranking: ListMLE-style loss ~200
    - Embedding: InfoNCE-style loss ~0.05
    """

    def __init__(self, d_model: int = 128):
        super().__init__()
        self.backbone = SharedTransformerBackbone(d_model=d_model)
        self.head_regression = nn.Linear(d_model, 1)
        self.head_classification = nn.Linear(d_model, 1)
        self.head_ranking = nn.Linear(d_model, 1)
        self.head_embedding = nn.Linear(d_model, 32)

    def forward(self, x):
        h = self.backbone(x)  # (B, T, D)
        h_pool = h.mean(dim=1)  # (B, D)
        return (
            self.head_regression(h_pool),
            self.head_classification(h_pool),
            self.head_ranking(h_pool),
            self.head_embedding(h_pool),
        )


def info_nce_loss(embeddings, temperature=0.1):
    """Simplified InfoNCE contrastive loss."""
    embeddings = nn.functional.normalize(embeddings, dim=1)
    sim = embeddings @ embeddings.T / temperature
    labels = torch.arange(len(embeddings), device=embeddings.device)
    return nn.functional.cross_entropy(sim, labels)


def main():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MultiHeadModel(d_model=128).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    # Golden Pendulum MTL — matches paper's 4-task setup
    balancer = GoldenPendulumMTL(n_tasks=4, lam=0.5)
    weight_logger = WeightLogger(log_file="weight_history.jsonl", log_every=50)

    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Golden-ratio targets for K=4: (0.106, 0.171, 0.276, 0.447)")
    print(f"Device: {device}\n")

    batch_size, seq_len, d_in = 32, 32, 16
    n_steps = 500

    for step in range(n_steps):
        optimizer.zero_grad()

        x = torch.randn(batch_size, seq_len, d_in, device=device)
        y_reg = torch.randn(batch_size, 1, device=device) * 0.01  # Small scale
        y_cls = torch.randint(0, 2, (batch_size, 1), device=device).float()
        y_rank = torch.randn(batch_size, 1, device=device) * 100  # Large scale

        reg_out, cls_out, rank_out, embed_out = model(x)

        losses = {
            "horizon_returns": nn.functional.mse_loss(reg_out, y_reg),  # ~0.01
            "trade_quality": nn.functional.binary_cross_entropy_with_logits(cls_out, y_cls),  # ~0.69
            "ranking": nn.functional.mse_loss(rank_out, y_rank),  # ~200
            "embedding": info_nce_loss(embed_out),  # ~0.05
        }

        weights = balancer.backward(losses, model)
        optimizer.step()

        weight_logger.log(step, weights)

        if step % 100 == 0:
            loss_str = " | ".join(f"{k}={v.item():.4f}" for k, v in losses.items())
            print(f"Step {step:4d} | {loss_str}")
            print(f"         | weights: {', '.join(f'{k}={v:.4f}' for k, v in weights.items())}")
            print(f"         | balance: {balancer.weight_balance_ratio:.3f}\n")

    # Final summary
    print("=" * 60)
    print("Training Complete")
    print(f"Final balance ratio: {balancer.weight_balance_ratio:.3f}")
    print(f"  (Nash-MTL typically gets 0.04, Golden Pendulum targets > 0.20)")
    print(f"\nMean weights (last 100 steps):")
    for k, v in balancer.mean_weights(100).items():
        print(f"  {k}: {v:.4f}")
    print(f"\nGolden targets:")
    for k, v in balancer.golden_targets.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
