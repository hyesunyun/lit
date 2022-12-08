"""Wrapper for HuggingFace models in LIT.

Includes BERT masked LM, GPT-2, and T5.

This wrapper loads a model into memory and implements the a number of helper
functions to predict a batch of examples and extract information such as
hidden states and attention.
"""
import re
from typing import Dict, List, Tuple

from lit_nlp.api import model as lit_model
from lit_nlp.api import types as lit_types
from lit_nlp.examples.models import model_utils
from lit_nlp.lib import utils

import numpy as np
import tensorflow as tf
import transformers
import torch

from bert_score import BERTScorer


class BertMLM(lit_model.Model):
  """BERT masked LM using Huggingface Transformers and TensorFlow 2."""

  MASK_TOKEN = "[MASK]"

  @property
  def max_seq_length(self):
    return self.model.config.max_position_embeddings

  def __init__(self, model_name="bert-base-uncased", top_k=10):
    super().__init__()
    self.tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, use_fast=False)
    # TODO(lit-dev): switch to TFBertForPreTraining to get the next-sentence
    # prediction head as well.
    self.model = model_utils.load_pretrained(
        transformers.TFBertForMaskedLM,
        model_name,
        output_hidden_states=True,
        output_attentions=True)
    self.top_k = top_k

  # TODO(lit-dev): break this out as a helper function, write some tests,
  # and de-duplicate code with the other text generation functions.
  def _get_topk_tokens(self,
                       scores: np.ndarray) -> List[List[Tuple[str, float]]]:
    """Convert raw scores to top-k token predictions."""
    # scores is [num_tokens, vocab_size]
    # Find the vocab indices of top k predictions, at each token.
    # np.argpartition is faster than a full argsort for k << V,
    # but we need to sort the output after slicing (see below).
    index_array = np.argpartition(scores, -self.top_k, axis=1)[:, -self.top_k:]
    # These are each [num_tokens, tok_k]
    top_tokens = [
        self.tokenizer.convert_ids_to_tokens(idxs) for idxs in index_array
    ]
    top_scores = np.take_along_axis(scores, index_array, axis=1)
    # Convert to a list of lists of (token, score) pairs,
    # where inner lists are sorted in descending order of score.
    return [
        sorted(list(zip(toks, scores)), key=lambda ab: -ab[1])
        for toks, scores in zip(top_tokens, top_scores)
    ]
    # TODO(lit-dev): consider returning indices and a vocab, since repeating
    # strings is slow and redundant.

  def _postprocess(self, output: Dict[str, np.ndarray]):
    """Postprocess, modifying output dict in-place."""
    # Slice to remove padding, omitting initial [CLS] and final [SEP]
    slicer = slice(1, output.pop("ntok") - 1)
    output["tokens"] = self.tokenizer.convert_ids_to_tokens(
        output.pop("input_ids")[slicer])
    probas = output.pop("probas")

    # Predictions at every position, regardless of masking.
    output["pred_tokens"] = self._get_topk_tokens(probas[slicer])  # pytype: disable=container-type-mismatch

    return output

  ##
  # LIT API implementations
  def max_minibatch_size(self) -> int:
    # The lit.Model base class handles batching automatically in the
    # implementation of predict(), and uses this value as the batch size.
    return 8

  def predict_minibatch(self, inputs):
    """Predict on a single minibatch of examples."""
    # If input has a 'tokens' field, use that. Otherwise tokenize the text.
    tokenized_texts = [
        ex.get("tokens") or self.tokenizer.tokenize(ex["text"]) for ex in inputs
    ]
    encoded_input = model_utils.batch_encode_pretokenized(
        self.tokenizer, tokenized_texts)

    # out.logits is a single tensor
    #    <float32>[batch_size, num_tokens, vocab_size]
    # out.hidden_states is a list of num_layers + 1 tensors, each
    #    <float32>[batch_size, num_tokens, h_dim]
    out: transformers.modeling_tf_outputs.TFMaskedLMOutput = \
        self.model(encoded_input)
    batched_outputs = {
        "probas": tf.nn.softmax(out.logits, axis=-1).numpy(),
        "input_ids": encoded_input["input_ids"].numpy(),
        "ntok": tf.reduce_sum(encoded_input["attention_mask"], axis=1).numpy(),
        # last layer, first token
        "cls_emb": out.hidden_states[-1][:, 0].numpy(),
    }
    # List of dicts, one per example.
    unbatched_outputs = utils.unbatch_preds(batched_outputs)
    # Postprocess to remove padding and decode predictions.
    return map(self._postprocess, unbatched_outputs)

  def load(self, model_name_or_path):
    """Dynamically load a new BertMLM model given a model name."""
    return BertMLM(model_name_or_path, self.top_k)

  def input_spec(self):
    return {
        "text": lit_types.TextSegment(),
        "tokens": lit_types.Tokens(mask_token="[MASK]", required=False),
    }

  def output_spec(self):
    return {
        "tokens": lit_types.Tokens(parent="text"),
        "pred_tokens": lit_types.TokenTopKPreds(align="tokens"),
        "cls_emb": lit_types.Embeddings(),
    }


class GPT2LanguageModel(lit_model.Model):
  """Wrapper for a Huggingface Transformers GPT-2 model.

  This class loads a tokenizer and model using the Huggingface library and
  provides the LIT-required functions plus additional helper functions to
  convert and clean tokens and to compute the top_k predictions from logits.
  """

  @property
  def num_layers(self):
    return self.model.config.n_layer

  def __init__(self, model_name="gpt2", **config_kw):
    """Constructor for GPT2LanguageModel.

    Args:
      model_name: gpt2, gpt2-medium, gpt2-large, gpt2-xl, distilgpt2, etc.
      top_k: How many predictions to prune.
    """
    super().__init__()
    assert self.config.num_to_generate <= self.config.beam_size
    # GPT2 is trained without pad_token, so pick arbitrary one and mask out.
    self.tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, pad_token="<|endoftext|>", use_fast=False)
    self.model = transformers.TFGPT2LMHeadModel.from_pretrained(
        model_name, output_hidden_states=True, output_attentions=True)

  def _encode_texts(self, texts: List[str]):
    return self.tokenizer.batch_encode_plus(
        texts,
        return_tensors="tf",
        add_special_tokens=True,
        padding="longest",
        truncation="longest_first")

  def _force_decode(self, encoded_inputs, encoded_targets):
    """Get predictions for a batch of tokenized examples.
    Each forward pass produces the following:
      logits: batch_size x dec_len x vocab_size
      decoder_past_key_value_states: tuple with cached outputs.
      dec_states: tuple[len:dec_layers]:
                  batch_size x dec_len x hid_size
      dec_attn: [optional] tuple[len:dec_layers+1]
                batch_size x num_heads x dec_len x dec_len
      enc_final_state: batch_size x enc_len x hid_size
      enc_states: tuple[len:enc_layers]:
                  batch_size x enc_len x hid_size
      enc_attn: [optional] tuple[len:enc_layers+1]
                batch_size x num_heads x enc_len x enc_len
    The two optional attention fields are only returned if
    config.output_attention is set.
    Args:
      encoded_inputs: Dict as returned from Tokenizer for inputs.
      encoded_targets: Dict as returned from Tokenizer for outputs
    Returns:
      batched_outputs: Dict[str, tf.Tensor]
    """
    results = self.model(
        input_ids=encoded_inputs["input_ids"],
        decoder_input_ids=encoded_targets["input_ids"],
        attention_mask=encoded_inputs["attention_mask"],
        decoder_attention_mask=encoded_targets["attention_mask"])

    model_probs = tf.nn.softmax(results.logits, axis=-1)
    top_k = tf.math.top_k(
        model_probs, k=self.config.token_top_k, sorted=True, name=None)
    batched_outputs = {
        "input_ids": encoded_inputs["input_ids"],
        "input_ntok": tf.reduce_sum(encoded_inputs["attention_mask"], axis=1),
        "target_ids": encoded_targets["input_ids"],
        "target_ntok": tf.reduce_sum(encoded_targets["attention_mask"], axis=1),
        "top_k_indices": top_k.indices,
        "top_k_probs": top_k.values,
    }
    # encoder_last_hidden_state is <float>[batch_size, num_tokens, emb_dim]
    # take the mean over real tokens to get <float>[batch_size, emb_dim]
    batched_outputs["encoder_final_embedding"] = masked_token_mean(
        results.encoder_last_hidden_state, encoded_inputs["attention_mask"])

    if self.config.output_attention:
      for i in range(len(results.decoder_attentions)):
        batched_outputs[
            f"decoder_layer_{i+1:d}_attention"] = results.decoder_attentions[i]
      for i in range(len(results.encoder_attentions)):
        batched_outputs[
            f"encoder_layer_{i+1:d}_attention"] = results.encoder_attentions[i]

    return batched_outputs

  @staticmethod
  def clean_bpe_token(tok):
    if not tok.startswith("Ġ"):
      return "_" + tok
    else:
      return tok.replace("Ġ", "")

  def _detokenize(self, ids):
    tokens = self.tokenizer.convert_ids_to_tokens(ids)
    return [self.clean_bpe_token(t) for t in tokens]

  def _postprocess(self, preds):
    """Post-process single-example preds. Operates on numpy arrays."""
    # Return tokenization for input text.
    input_ntok = preds.pop("input_ntok")
    input_ids = preds.pop("input_ids")[:input_ntok]
    preds["input_tokens"] = self._detokenize(input_ids)
    # Return tokenization for target text.
    target_ntok = preds.pop("target_ntok")
    target_ids = preds.pop("target_ids")[:target_ntok]
    preds["target_tokens"] = self._detokenize(target_ids)

    # Decode predicted top-k tokens.
    # token_topk_preds will be a List[List[(word, prob)]]
    # Initialize prediction for 0th token as N/A.
    token_topk_preds = [[("N/A", 1.)]]
    pred_ids = preds.pop("top_k_indices")[:target_ntok]  # <int>[num_tokens, k]
    pred_probs = preds.pop(
        "top_k_probs")[:target_ntok]  # <float32>[num_tokens, k]
    for token_pred_ids, token_pred_probs in zip(pred_ids, pred_probs):
      token_pred_words = self._detokenize(token_pred_ids)
      token_topk_preds.append(list(zip(token_pred_words, token_pred_probs)))
    preds["pred_tokens"] = token_topk_preds

    # Decode generated ids
    candidates = [
        self.tokenizer.decode(ids, skip_special_tokens=True)
        for ids in preds.pop("generated_ids")
    ]
    if self.config.num_to_generate > 1:
      preds["output_text"] = [(s, None) for s in candidates]
    else:
      preds["output_text"] = candidates[0]

    # Process attention fields, if present.
    for key in preds:
      if not re.match(r"\w+_layer_(\d+)/attention", key):
        continue
      if key.startswith("encoder_"):
        ntok = input_ntok
      elif key.startswith("decoder_"):
        ntok = target_ntok
      else:
        raise ValueError(f"Invalid attention key: '{key}'")
      # Select only real tokens, since most of this matrix is padding.
      # <float32>[num_heads, max_seq_length, max_seq_length]
      # -> <float32>[num_heads, num_tokens, num_tokens]
      preds[key] = preds[key][:, :ntok, :ntok].transpose((0, 2, 1))
      # Make a copy of this array to avoid memory leaks, since NumPy otherwise
      # keeps a pointer around that prevents the source array from being GCed.
      preds[key] = preds[key].copy()

    return preds

  ##
  # LIT API implementations
  def max_minibatch_size(self) -> int:
    # The lit.Model base class handles batching automatically in the
    # implementation of predict(), and uses this value as the batch size.
    return 6

  def predict_minibatch(self, inputs):
    """Run model on a single batch.
    Args:
      inputs: List[Dict] with fields as described by input_spec()
    Returns:
      outputs: List[Dict] with fields as described by output_spec()
    """
    # Text as sequence of sentencepiece ID"s.
    encoded_inputs = self._encode_texts([ex["input_text"] for ex in inputs])
    encoded_targets = self._encode_texts(
        [ex.get("target_text", "") for ex in inputs])

    ##
    # Force-decode on target text, and also get encoder embs and attention.
    batched_outputs = self._force_decode(encoded_inputs, encoded_targets)
    # Get the conditional generation from the model.
    # Workaround for output_hidden not being compatible with generate.
    # See https://github.com/huggingface/transformers/issues/8361
    self.model.config.output_hidden_states = False
    generated_ids = self.model.generate(
        encoded_inputs.input_ids,
        num_beams=self.config.beam_size,
        attention_mask=encoded_inputs.attention_mask,
        max_length=self.config.max_gen_length,
        num_return_sequences=self.config.num_to_generate)
    # [batch_size*num_return_sequences, num_steps]
    # -> [batch_size, num_return_sequences, num_steps]
    batched_outputs["generated_ids"] = tf.reshape(
        generated_ids,
        [-1, self.config.num_to_generate, generated_ids.shape[-1]])
    self.model.config.output_hidden_states = True

    # Convert to numpy for post-processing.
    detached_outputs = {k: v.numpy() for k, v in batched_outputs.items()}
    # Split up batched outputs, then post-process each example.
    unbatched_outputs = utils.unbatch_preds(detached_outputs)
    return list(map(self._postprocess, unbatched_outputs))

  def input_spec(self):
    return {
        "input_text": lit_types.TextSegment(),
        "target_text": lit_types.TextSegment(required=False),
    }

  def output_spec(self):
    spec = {
        "output_text": lit_types.GeneratedText(parent="target_text"),
        "input_tokens": lit_types.Tokens(parent="input_text"),
        "encoder_final_embedding": lit_types.Embeddings(),
        # If target text is given, the following will also be populated.
        "target_tokens": lit_types.Tokens(parent="target_text"),
        "pred_tokens": lit_types.TokenTopKPreds(align="target_tokens"),
    }
    if self.config.num_to_generate > 1:
      spec["output_text"] = lit_types.GeneratedTextCandidates(
          parent="target_text")

    if self.config.output_attention:
      # Add attention for each layer.
      for i in range(self.num_layers):
        spec[f"encoder_layer_{i+1:d}_attention"] = lit_types.AttentionHeads(
            align_in="input_tokens", align_out="input_tokens")
        spec[f"decoder_layer_{i+1:d}_attention"] = lit_types.AttentionHeads(
            align_in="target_tokens", align_out="target_tokens")
    return spec

class BioGPTLanguageModel(lit_model.Model):
  """Wrapper for a Huggingface Transformers BioGPTLanguageModel model.

  This class loads a tokenizer and model using the Huggingface library and
  provides the LIT-required functions plus additional helper functions to
  convert and clean tokens and to compute the top_k predictions from logits.
  """

  @property
  def num_layers(self):
    return self.model.config.num_hidden_layers

  def __init__(self, model_name="microsoft/biogpt", top_k=10):
    """Constructor for BioGPTLanguageModel.

    Args:
      model_name: microsoft/biogpt
      top_k: How many predictions to prune.
    """
    super().__init__()
    # GPT2-medium/BioGPT is trained without pad_token, so pick arbitrary one and mask out.
    self.tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, pad_token="<|endoftext|>", use_fast=False)
    self.model = transformers.BioGptForCausalLM.from_pretrained(
        model_name, output_hidden_states=True, output_attentions=True)
    self.top_k = top_k

  def _encode_texts(self, texts: List[str]):
    return self.tokenizer.batch_encode_plus(
        texts,
        return_tensors="pt",
        add_special_tokens=True,
        padding="longest",
        truncation="longest_first")

  def _force_decode(self, encoded_inputs, encoded_targets):
    """Get predictions for a batch of tokenized examples.
    Each forward pass produces the following:
      logits: batch_size x dec_len x vocab_size
      decoder_past_key_value_states: tuple with cached outputs.
      dec_states: tuple[len:dec_layers]:
                  batch_size x dec_len x hid_size
      dec_attn: [optional] tuple[len:dec_layers+1]
                batch_size x num_heads x dec_len x dec_len
      enc_final_state: batch_size x enc_len x hid_size
      enc_states: tuple[len:enc_layers]:
                  batch_size x enc_len x hid_size
      enc_attn: [optional] tuple[len:enc_layers+1]
                batch_size x num_heads x enc_len x enc_len
    The two optional attention fields are only returned if
    config.output_attention is set.
    Args:
      encoded_inputs: Dict as returned from Tokenizer for inputs.
      encoded_targets: Dict as returned from Tokenizer for outputs
    Returns:
      batched_outputs: Dict[str, torch.Tensor]
    """
    results = self.model(
        input_ids=encoded_inputs["input_ids"],
        decoder_input_ids=encoded_targets["input_ids"],
        attention_mask=encoded_inputs["attention_mask"],
        decoder_attention_mask=encoded_targets["attention_mask"])

    model_probs = torch.softmax(results.logits, axis=-1, dtype=torch.float32)
    top_k = torch.topk(
        model_probs, k=self.config.token_top_k, sorted=True)
    batched_outputs = {
        "input_ids": encoded_inputs["input_ids"],
        "input_ntok": torch.sum(encoded_inputs["attention_mask"], axis=1),
        "target_ids": encoded_targets["input_ids"],
        "target_ntok": torch.sum(encoded_targets["attention_mask"], axis=1),
        "top_k_indices": top_k.indices,
        "top_k_probs": top_k.values,
    }
    # encoder_last_hidden_state is <float>[batch_size, num_tokens, emb_dim]
    # take the mean over real tokens to get <float>[batch_size, emb_dim]
    batched_outputs["encoder_final_embedding"] = masked_token_mean(
        results.encoder_last_hidden_state, encoded_inputs["attention_mask"])

    if self.config.output_attention:
      for i in range(len(results.decoder_attentions)):
        batched_outputs[
            f"decoder_layer_{i+1:d}_attention"] = results.decoder_attentions[i]
      for i in range(len(results.encoder_attentions)):
        batched_outputs[
            f"encoder_layer_{i+1:d}_attention"] = results.encoder_attentions[i]

    return batched_outputs

  @staticmethod
  def clean_bpe_token(tok):
    if not tok.startswith("Ġ"):
      return "_" + tok
    else:
      return tok.replace("Ġ", "")

  def _detokenize(self, ids):
    tokens = self.tokenizer.convert_ids_to_tokens(ids)
    return [self.clean_bpe_token(t) for t in tokens]

  def _postprocess(self, preds):
    """Post-process single-example preds. Operates on numpy arrays."""
    # Return tokenization for input text.
    input_ntok = preds.pop("input_ntok")
    input_ids = preds.pop("input_ids")[:input_ntok]
    preds["input_tokens"] = self._detokenize(input_ids)
    # Return tokenization for target text.
    target_ntok = preds.pop("target_ntok")
    target_ids = preds.pop("target_ids")[:target_ntok]
    preds["target_tokens"] = self._detokenize(target_ids)

    # Decode predicted top-k tokens.
    # token_topk_preds will be a List[List[(word, prob)]]
    # Initialize prediction for 0th token as N/A.
    token_topk_preds = [[("N/A", 1.)]]
    pred_ids = preds.pop("top_k_indices")[:target_ntok]  # <int>[num_tokens, k]
    pred_probs = preds.pop(
        "top_k_probs")[:target_ntok]  # <float32>[num_tokens, k]
    for token_pred_ids, token_pred_probs in zip(pred_ids, pred_probs):
      token_pred_words = self._detokenize(token_pred_ids)
      token_topk_preds.append(list(zip(token_pred_words, token_pred_probs)))
    preds["pred_tokens"] = token_topk_preds

    # Decode generated ids
    candidates = [
        self.tokenizer.decode(ids, skip_special_tokens=True)
        for ids in preds.pop("generated_ids")
    ]
    if self.config.num_to_generate > 1:
      preds["output_text"] = [(s, None) for s in candidates]
    else:
      preds["output_text"] = candidates[0]

    # Process attention fields, if present.
    for key in preds:
      if not re.match(r"\w+_layer_(\d+)/attention", key):
        continue
      if key.startswith("encoder_"):
        ntok = input_ntok
      elif key.startswith("decoder_"):
        ntok = target_ntok
      else:
        raise ValueError(f"Invalid attention key: '{key}'")
      # Select only real tokens, since most of this matrix is padding.
      # <float32>[num_heads, max_seq_length, max_seq_length]
      # -> <float32>[num_heads, num_tokens, num_tokens]
      preds[key] = preds[key][:, :ntok, :ntok].transpose((0, 2, 1))
      # Make a copy of this array to avoid memory leaks, since NumPy otherwise
      # keeps a pointer around that prevents the source array from being GCed.
      preds[key] = preds[key].copy()

    return preds

  ##
  # LIT API implementations
  def max_minibatch_size(self) -> int:
    # The lit.Model base class handles batching automatically in the
    # implementation of predict(), and uses this value as the batch size.
    return 6

  def predict_minibatch(self, inputs):
    """Run model on a single batch.
    Args:
      inputs: List[Dict] with fields as described by input_spec()
    Returns:
      outputs: List[Dict] with fields as described by output_spec()
    """
    # Text as sequence of sentencepiece ID"s.
    encoded_inputs = self._encode_texts([ex["input_text"] for ex in inputs])
    encoded_targets = self._encode_texts(
        [ex.get("target_text", "") for ex in inputs])

    ##
    # Force-decode on target text, and also get encoder embs and attention.
    batched_outputs = self._force_decode(encoded_inputs, encoded_targets)
    # Get the conditional generation from the model.
    # Workaround for output_hidden not being compatible with generate.
    # See https://github.com/huggingface/transformers/issues/8361
    self.model.config.output_hidden_states = False
    generated_ids = self.model.generate(
        encoded_inputs.input_ids,
        num_beams=self.config.beam_size,
        attention_mask=encoded_inputs.attention_mask,
        max_length=self.config.max_gen_length,
        num_return_sequences=self.config.num_to_generate)
    # [batch_size*num_return_sequences, num_steps]
    # -> [batch_size, num_return_sequences, num_steps]
    batched_outputs["generated_ids"] = torch.reshape(
        generated_ids,
        (-1, self.config.num_to_generate, generated_ids.shape[-1])
      )
    self.model.config.output_hidden_states = True

    # Convert to numpy for post-processing.
    detached_outputs = {k: v.numpy() for k, v in batched_outputs.items()}
    # Split up batched outputs, then post-process each example.
    unbatched_outputs = utils.unbatch_preds(detached_outputs)
    return list(map(self._postprocess, unbatched_outputs))

  def input_spec(self):
    return {
        "input_text": lit_types.TextSegment(),
        "target_text": lit_types.TextSegment(required=False),
    }

  def output_spec(self):
    spec = {
        "output_text": lit_types.GeneratedText(parent="target_text"),
        "input_tokens": lit_types.Tokens(parent="input_text"),
        "encoder_final_embedding": lit_types.Embeddings(),
        # If target text is given, the following will also be populated.
        "target_tokens": lit_types.Tokens(parent="target_text"),
        "pred_tokens": lit_types.TokenTopKPreds(align="target_tokens"),
    }
    if self.config.num_to_generate > 1:
      spec["output_text"] = lit_types.GeneratedTextCandidates(
          parent="target_text")

    if self.config.output_attention:
      # Add attention for each layer.
      for i in range(self.num_layers):
        spec[f"encoder_layer_{i+1:d}_attention"] = lit_types.AttentionHeads(
            align_in="input_tokens", align_out="input_tokens")
        spec[f"decoder_layer_{i+1:d}_attention"] = lit_types.AttentionHeads(
            align_in="target_tokens", align_out="target_tokens")
    return spec

class SafeTextDecisionWrapper(lit_model.ModelWrapper):
  """Wrapper class to perform a safe text benchmark decision task."""

  # Mapping from generic SafeText fields to this task
  FIELD_RENAMES = {
      "input_text": "text",
      "target_text": "target",
  }

  def __init__(self, model: lit_model.Model):
    super().__init__(model)

    # TODO(gehrmann): temp solution for BertScorer.
    self._scorer = scorer = BERTScorer(lang="en", rescale_with_baseline=True)
    # If output is List[(str, score)] instead of just str
    self._multi_output = isinstance(self.output_spec()["output_text"],
                                    lit_types.GeneratedTextCandidates)
    self._get_pred_string = (
        lit_types.GeneratedTextCandidates.top_text if self._multi_output else
        (lambda x: x))

  def preprocess(self, ex: JsonDict) -> JsonDict:
    ret = {"input_text": ex["text"]}
    if "target" in ex:
      ret["target_text"] = ex["target"]
    return ret

  ##
  # LIT API implementation
  def description(self) -> str:
    return "For safe text decision\n" + self.wrapped.description()

  # TODO(b/170662608): remove these after batching API is cleaned up.
  def max_minibatch_size(self) -> int:
    raise NotImplementedError("Use predict() instead.")

  def predict_minibatch(self, inputs):
    raise NotImplementedError("Use predict() instead.")

  def predict(self, inputs):
    """Predict on a single minibatch of examples."""
    inputs = list(inputs)  # needs to be referenced below, so keep full list
    model_inputs = (self.preprocess(ex) for ex in inputs)
    outputs = self.wrapped.predict(model_inputs)
    outputs = (utils.remap_dict(mo, self.FIELD_RENAMES) for mo in outputs)

    # TODO(gehrmann): temp solution to get ROUGE scores in data table.
    for ex, mo in zip(inputs, outputs):
      P, R, F1 = self._scorer.score(
          self._get_pred_string(mo["output_text"],
          ex["target"]))
      mo["BERTScore"] = float(F1[0])
      yield mo

  def predict_with_metadata(self, indexed_inputs):
    """As predict(), but inputs are IndexedInput."""
    return self.predict((ex["data"] for ex in indexed_inputs))

  def input_spec(self):
    return lit_types.remap_spec(self.wrapped.input_spec(), self.FIELD_RENAMES)

  def output_spec(self):
    spec = lit_types.remap_spec(self.wrapped.output_spec(), self.FIELD_RENAMES)
    spec["BERTScore"] = lit_types.Scalar()
    return spec

class SafeTextGenerationWrapper(lit_model.ModelWrapper):
  """Wrapper class to perform a safe text benchmark generation task."""

  # Mapping from generic SafeText fields to this task
  FIELD_RENAMES = {
      "input_text": "text",
      "target_text": "safe_advices",
  }

  def __init__(self, model: lit_model.Model):
    super().__init__(model)

    # TODO(gehrmann): temp solution for BertScorer.
    self._scorer = scorer = BERTScorer(lang="en", rescale_with_baseline=True)
    # If output is List[(str, score)] instead of just str
    self._multi_output = isinstance(self.output_spec()["output_text"],
                                    lit_types.GeneratedTextCandidates)
    self._get_pred_string = (
        lit_types.GeneratedTextCandidates.top_text if self._multi_output else
        (lambda x: x))

  def preprocess(self, ex: JsonDict) -> JsonDict:
    ret = {"input_text": ex["text"]}
    if "target" in ex:
      ret["target_text"] = ex["safe_advices"]
    return ret

  ##
  # LIT API implementation
  def description(self) -> str:
    return "For safe text generation\n" + self.wrapped.description()

  # TODO(b/170662608): remove these after batching API is cleaned up.
  def max_minibatch_size(self) -> int:
    raise NotImplementedError("Use predict() instead.")

  def predict_minibatch(self, inputs):
    raise NotImplementedError("Use predict() instead.")

  def predict(self, inputs):
    """Predict on a single minibatch of examples."""
    inputs = list(inputs)  # needs to be referenced below, so keep full list
    model_inputs = (self.preprocess(ex) for ex in inputs)
    outputs = self.wrapped.predict(model_inputs)
    outputs = (utils.remap_dict(mo, self.FIELD_RENAMES) for mo in outputs)

    # TODO(gehrmann): temp solution to get ROUGE scores in data table.
    for ex, mo in zip(inputs, outputs):
      P, R, F1 = self._scorer.score(
          self._get_pred_string(mo["output_text"],
          ex["safe_advices"]))
      mo["BERTScore"] = float(F1[0])
      yield mo

  def predict_with_metadata(self, indexed_inputs):
    """As predict(), but inputs are IndexedInput."""
    return self.predict((ex["data"] for ex in indexed_inputs))

  def input_spec(self):
    return lit_types.remap_spec(self.wrapped.input_spec(), self.FIELD_RENAMES)

  def output_spec(self):
    spec = lit_types.remap_spec(self.wrapped.output_spec(), self.FIELD_RENAMES)
    spec["BERTScore"] = lit_types.Scalar()
    return spec
