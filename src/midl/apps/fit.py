from __future__ import annotations

from collections.abc import Sequence

from absl import app

from midl.apps import bootstrap
from midl.cohort import make_dataloaders
from midl.fitting.loop import Trainer


def main(argv: Sequence[str]) -> None:
    del argv
    log = bootstrap.configure()
    loaders = make_dataloaders()
    trainer = Trainer()
    result = trainer.fit(loaders)
    log.info("training complete best_val_auroc=%.4f", result["best_val_auroc"])


def run() -> None:
    app.run(main)


if __name__ == "__main__":
    run()
