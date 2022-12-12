# Lint as: python3
r"""Example demo loading pre-trained language models.

Currently supports the following model types:
- GPT-2 (gpt2* or distilgpt2) as a left-to-right language model
- BioGPT trained on biomedical text from GPT-2 medium

To run locally:
  python -m lit_nlp.examples.lm_safety_demo \
      --models=gpt2-medium --top_k 10 --port=5432 --max_examples=50 --alsologtostderr

To run using remote host:
  python -m lit_nlp.examples.lm_safety_demo --top_k=10 --port=5432 --max_examples=100 --alsologtostderr --host=0.0.0.0

Then navigate to localhost:5432 to access the demo UI.
"""
import os
import sys

from absl import app
from absl import flags
from absl import logging

from typing import Optional, Sequence

from lit_nlp import dev_server
from lit_nlp import server_flags
from lit_nlp.api import layout
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.components import word_replacer
from lit_nlp.examples.datasets import lm
from lit_nlp.examples.models import pretrained_lms

# NOTE: additional flags defined in server_flags.py

_MODELS = flags.DEFINE_list(
    "models", ["gpt2-medium", "microsoft/biogpt"],
    "Models to load. Currently supports GPT-2 medium and BioGPT.")

_TOKEN_TOP_K = flags.DEFINE_integer("top_k", 10,
                     "Rank to which the output distribution is pruned.")

_MAX_EXAMPLES = flags.DEFINE_integer(
    "max_examples", 500,
    "Maximum number of examples to load from each evaluation set. Set to None to load the full set."
)

_NUM_TO_GEN = flags.DEFINE_integer(
    "num_to_generate", 1, "Number of generations to produce for each input.")

_TASKS = flags.DEFINE_list("tasks", ["decision", "generation"],
                           "Which task(s) to load.")

# Custom frontend layout; see api/layout.py
modules = layout.LitModuleName
LM_LAYOUT = layout.LitCanonicalLayout(
    upper={
        "Main": [
            modules.EmbeddingsModule,
            modules.DataTableModule,
            modules.DatapointEditorModule,
        ]
    },
    lower={
        "Predictions": [
            modules.LanguageModelPredictionModule,
            modules.ConfusionMatrixModule,
        ],
        "Counterfactuals": [modules.GeneratorModule],
    },
    description="Custom layout for language models.",
)
CUSTOM_LAYOUTS = {"lm": LM_LAYOUT}

# You can also change this via URL param e.g. localhost:5432/?layout=default
FLAGS = flags.FLAGS

FLAGS.set_default("development_demo", True)


def get_wsgi_app():
  FLAGS.set_default("server_type", "external")
  FLAGS.set_default("demo_mode", True)
  # Parse flags without calling app.run(main), to avoid conflict with
  # gunicorn command line flags.
  unused = flags.FLAGS(sys.argv, known_only=True)
  return main(unused)


def main(argv: Sequence[str]) -> Optional[dev_server.LitServerType]:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  ##
  # Load models, according to the --models flag.
  base_models = {}
  for model_name_or_path in _MODELS.value:
    # Ignore path prefix, if using /path/to/<model_name> to load from a
    # specific directory rather than the default shortcut.
    model_name = os.path.basename(model_name_or_path)
    if model_name.startswith("gpt2"):
      base_models[model_name] = pretrained_lms.GPT2LanguageModel(
          model_name_or_path, token_top_k=_TOKEN_TOP_K.value, num_to_generate=_NUM_TO_GEN.value)
    elif model_name.startswith("biogpt"):
      base_models[model_name] = pretrained_lms.BioGPTLanguageModel(
          "microsoft/biogpt", token_top_k=_TOKEN_TOP_K.value, num_to_generate=_NUM_TO_GEN.value)
    else:
      raise ValueError(
          f"Unsupported model name '{model_name}' from path '{model_name_or_path}'"
      )

  models = {}
  datasets = {}

  if "decision" in _TASKS.value:
    for k, m in base_models.items():
      models[k + "_decision"] = pretrained_lms.SafeTextDecisionWrapper(m)
    datasets["safe-text-decision"] = lm.SafeTextDecisionData("/home/yun.hy/lit/lit_nlp/examples/datasets/SafeText/safetext_for_decision.csv")

  if "generation" in _TASKS.value:
    for k, m in base_models.items():
      models[k + "_generation"] = pretrained_lms.SafeTextGenerationWrapper(m)
    datasets["safe-text-gen"] = lm.SafeTextGenerationData("/home/yun.hy/lit/lit_nlp/examples/datasets/SafeText/safetext_for_generation.csv")

  for name in datasets:
    logging.info("Dataset: '%s' with %d examples", name, len(datasets[name]))
    datasets[name] = datasets[name].slice[:_MAX_EXAMPLES.value]
    logging.info("  truncated to %d examples", len(datasets[name]))

  generators = {"word_replacer": word_replacer.WordReplacer()}

  lit_demo = dev_server.Server(
      models,
      datasets,
      generators=generators,
      layouts=CUSTOM_LAYOUTS,
      **server_flags.get_flags())
  return lit_demo.serve()


if __name__ == "__main__":
  app.run(main)
