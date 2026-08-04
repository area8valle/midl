from __future__ import annotations

from collections.abc import Sequence

from absl import app

from midl.apps import bootstrap
from midl.cohort import make_dataloaders
from midl.fitting.loop import Trainer
from midl.observables.gauges import evaluate_progression, evaluate_tkr


def main(argv: Sequence[str]) -> None:
    del argv
    log = bootstrap.configure()
    loaders = make_dataloaders()
    trainer = Trainer()
    if bootstrap.FLAGS.checkpoint:
        trainer.load_checkpoint(bootstrap.FLAGS.checkpoint)
    preds = trainer.predict(loaders["test"])
    progression = evaluate_progression(preds["prog_score"], preds["prog_label"])
    tkr = evaluate_tkr(preds["cox_risk"], preds["tkr_time"], preds["tkr_event"])
    log.info(
        "progression AUROC %.3f [%.3f, %.3f] brier %.3f auc_prc %.3f",
        progression["auroc"],
        progression["ci_low"],
        progression["ci_high"],
        progression["brier"],
        progression["auc_prc"],
    )
    log.info("TKR c-index %.3f auc_prc %.3f", tkr["c_index"], tkr["auc_prc"])


def run() -> None:
    app.run(main)


if __name__ == "__main__":
    run()
