from __future__ import annotations

import logging
import os

import gin
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_multi_string("gin_file", [], "Paths to gin configuration files.")
flags.DEFINE_multi_string("gin_param", [], "Individual gin parameter bindings.")
flags.DEFINE_string("checkpoint", "", "Checkpoint path to load.")
flags.DEFINE_string("output", "", "Output path for artifacts.")
flags.DEFINE_integer("samples", 64, "Sample count for synthetic generation or inference.")


def configure() -> logging.Logger:
    gin.add_config_file_search_path(os.getcwd())
    gin.parse_config_files_and_bindings(
        list(FLAGS.gin_file), list(FLAGS.gin_param), finalize_config=False
    )
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    return logging.getLogger("midl")
