from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from absl import app

from midl.apps import bootstrap
from midl.cohort import synthesize_cohort


def main(argv: Sequence[str]) -> None:
    del argv
    log = bootstrap.configure()
    cohort = synthesize_cohort(n=bootstrap.FLAGS.samples)
    path = bootstrap.FLAGS.output or "synthetic_cohort.npz"
    np.savez_compressed(
        path,
        images=cohort.images,
        clinical=cohort.clinical,
        progression=cohort.progression,
        tkr_time=cohort.tkr_time,
        tkr_event=cohort.tkr_event,
        kl_grade=cohort.kl_grade,
    )
    log.info(
        "wrote synthetic cohort of %d knees (progression rate %.3f, tkr rate %.3f) to %s",
        len(cohort),
        float(cohort.progression.mean()),
        float(cohort.tkr_event.mean()),
        path,
    )


def run() -> None:
    app.run(main)


if __name__ == "__main__":
    run()
