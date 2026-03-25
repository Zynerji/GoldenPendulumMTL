"""Example: Golden Pendulum MTL with PyTorch Lightning.

Demonstrates integration using the GoldenPendulumCallback.

Requires: pip install pytorch-lightning
"""

import torch
import torch.nn as nn

try:
    import pytorch_lightning as pl
except ImportError:
    print("Install pytorch-lightning: pip install pytorch-lightning")
    raise

from golden_pendulum import GoldenPendulumCallback


class MultiTaskLitModel(pl.LightningModule):
    """Multi-task model with Golden Pendulum gradient balancing."""

    def __init__(self, d_in: int = 16, d_model: int = 64, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False  # Required for custom backward

        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(d_in, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model), nn.ReLU(),
        )

        # Task heads
        self.head_a = nn.Linear(d_model, 1)  # Regression
        self.head_b = nn.Linear(d_model, 5)  # Classification
        self.head_c = nn.Linear(d_model, 1)  # Ranking (large loss)

        # Golden Pendulum callback
        self.golden = GoldenPendulumCallback(lam=0.5, log_every=50)

    def forward(self, x):
        h = self.backbone(x)
        return self.head_a(h), self.head_b(h), self.head_c(h)

    def training_step(self, batch, batch_idx):
        x, y_a, y_b, y_c = batch
        out_a, out_b, out_c = self(x)

        losses = {
            "regression": nn.functional.mse_loss(out_a.squeeze(), y_a),
            "classification": nn.functional.cross_entropy(out_b, y_b),
            "ranking": 100.0 * nn.functional.mse_loss(out_c.squeeze(), y_c),
        }

        opt = self.optimizers()
        opt.zero_grad()

        # Golden Pendulum backward replaces loss.backward()
        weights = self.golden.on_train_batch(losses, self, batch_idx)

        opt.step()

        # Log everything
        self.log("train/total_loss", sum(v.item() for v in losses.values()))
        self.log("train/balance", self.golden.balancer.weight_balance_ratio)
        for k, v in weights.items():
            self.log(f"train/w_{k}", v)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)


class SyntheticDataModule(pl.LightningDataModule):
    def __init__(self, n_samples=1000, batch_size=32):
        super().__init__()
        self.n_samples = n_samples
        self.batch_size = batch_size

    def setup(self, stage=None):
        x = torch.randn(self.n_samples, 16)
        y_a = torch.randn(self.n_samples)
        y_b = torch.randint(0, 5, (self.n_samples,))
        y_c = torch.randn(self.n_samples)
        self.dataset = torch.utils.data.TensorDataset(x, y_a, y_b, y_c)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)


if __name__ == "__main__":
    model = MultiTaskLitModel()
    data = SyntheticDataModule()
    trainer = pl.Trainer(max_epochs=5, accelerator="auto", log_every_n_steps=10)
    trainer.fit(model, data)
    print(f"\nFinal balance: {model.golden.balancer.weight_balance_ratio:.3f}")
    print(f"Mean weights: {model.golden.balancer.mean_weights()}")
