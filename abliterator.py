import torch
import torch.nn.functional as F
import functools
import einops
import gc
import re
from itertools import islice
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch import Tensor
from typing import Callable, Dict, List, Set, Tuple, Union
from transformer_lens import HookedTransformer, utils, ActivationCache
from transformer_lens.hook_points import HookPoint
from transformers import AutoTokenizer, AutoModelForCausalLM
from jaxtyping import Float, Int
import numpy as np
import pickle
import gzip
import logging


# Convert tensors to numpy arrays with float16 precision
def convert_tensors_to_numpy(cache: Dict[str, Tensor]) -> Dict[str, np.ndarray]:
    numpy_cache = {}
    for key, tensor in cache.items():
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.to(dtype=torch.float32)
        numpy_cache[key] = tensor.cpu().numpy().astype(np.float16)
    return numpy_cache


# Save the dictionary to a compressed file
def save_cache(cache: Dict[str, Tensor], file_name: str) -> None:
    numpy_cache = convert_tensors_to_numpy(cache)
    with gzip.open(file_name, "wb") as f:
        pickle.dump(numpy_cache, f)


# Load the dictionary from a compressed file and convert back to float32
def load_cache(file_name: str) -> Dict[str, Tensor]:
    with gzip.open(file_name, "rb") as f:
        numpy_cache = pickle.load(f)
    # Convert back to PyTorch tensors with float32 precision
    cache = {
        key: torch.tensor(array, dtype=torch.float32)
        for key, array in numpy_cache.items()
    }
    return cache


# Wrapper function to convert ActivationCache to numpy
def activation_cache_to_numpy(cache: ActivationCache) -> Dict[str, np.ndarray]:
    numpy_cache = {}
    for key, tensor in cache.items():
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.to(dtype=torch.float32)
        numpy_cache[key] = tensor.cpu().numpy().astype(np.float16)
    return numpy_cache


# Wrapper function to convert numpy back to ActivationCache
def numpy_to_activation_cache(
    numpy_cache: Dict[str, np.ndarray], model: HookedTransformer
) -> ActivationCache:
    cache = {
        key: torch.tensor(array, dtype=torch.bfloat16)
        for key, array in numpy_cache.items()
    }
    return ActivationCache(cache, model)


# Save the ActivationCache to a compressed file
def save_compressed_cache(cache: ActivationCache, file_name: str) -> None:
    numpy_cache = activation_cache_to_numpy(cache)
    with gzip.open(file_name, "wb") as f:
        pickle.dump(numpy_cache, f, protocol=pickle.HIGHEST_PROTOCOL)


# Load the ActivationCache from a compressed file
def load_compressed_cache(file_name: str, model: HookedTransformer) -> ActivationCache:
    with gzip.open(file_name, "rb") as f:
        numpy_cache = pickle.load(f)
    return numpy_to_activation_cache(numpy_cache, model)


def batch(iterable, n):
    it = iter(iterable)
    while True:
        chunk = list(islice(it, n))
        if not chunk:
            break
        yield chunk


def get_trait_instructions() -> Tuple[List[str], List[str]]:
    hf_path = "Undi95/orthogonal-activation-steering-TOXIC"
    dataset = load_dataset(hf_path)
    instructions = [i["goal"] for i in dataset["test"]]

    train, test = train_test_split(instructions, test_size=0.2, random_state=42)
    return train, test


def get_baseline_instructions() -> Tuple[List[str], List[str]]:
    hf_path = "tatsu-lab/alpaca"
    dataset = load_dataset(hf_path)
    instructions = [
        dataset["train"][i]["instruction"]
        for i in range(5000)
        if dataset["train"][i]["input"].strip() == ""
    ]
    train, test = train_test_split(instructions, test_size=0.2, random_state=42)
    return train, test


def prepare_dataset(
    dataset: Union[Tuple[List[str], List[str]], List[str]],
) -> Tuple[List[str], List[str]]:
    if len(dataset) != 2:
        train, test = train_test_split(dataset, test_size=0.1, random_state=42)
    else:
        train, test = dataset
    return train, test


def directional_hook(
    activation: Float[Tensor, "... d_model"],
    hook: HookPoint,
    direction: Float[Tensor, "d_model"],
) -> Float[Tensor, "... d_model"]:
    if activation.device != direction.device:
        direction = direction.to(activation.device)
    proj = (
        einops.einsum(
            activation,
            direction.view(-1, 1),
            "... d_model, d_model single -> ... single",
        )
        * direction
    )
    return activation - proj


def clear_mem():
    gc.collect()
    torch.cuda.empty_cache()


def measure_fn(
    measure: str, input_tensor: Tensor, *args, **kwargs
) -> Float[Tensor, "..."]:
    avail_measures = {
        "mean": torch.mean,
        "median": torch.median,
        "max": torch.max,
        "stack": torch.stack,
    }

    try:
        return avail_measures[measure](input_tensor, *args, **kwargs)
    except KeyError:
        raise NotImplementedError(
            f"Unknown measure function '{measure}'. Available measures:"
            + ", ".join([f"'{str(fn)}'" for fn in avail_measures.keys()])
        )


class ChatTemplate:
    def __init__(self, model, template: str):
        self.model = model
        self.template = template

    def format(self, instruction: str) -> str:
        return self.template.format(instruction=instruction)

    def __enter__(self):
        self.prev = self.model.chat_template
        self.model.chat_template = self
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.model.chat_template = self.prev
        del self.prev


LLAMA3_CHAT_TEMPLATE = """<|start_header_id|>user<|end_header_id|>\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"""
PHI3_CHAT_TEMPLATE = """<|user|>\n{instruction}<|end|>\n<|assistant|>"""


class ModelAbliterator:
    def __init__(
        self,
        model: str,
        dataset: Union[Tuple[List[str], List[str]], List[Tuple[List[str], List[str]]]],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        n_devices: int = None,
        local_files_only=False,
        cache_fname: str = None,
        activation_layers: List[str] = [
            "resid_pre",
            "resid_post",
            "mlp_out",
            "attn_out",
        ],
        chat_template: str = None,
        positive_toks: Union[
            List[int], Tuple[int], Set[int], Int[Tensor, "..."]
        ] = None,
        negative_toks: Union[
            List[int], Tuple[int], Set[int], Int[Tensor, "..."]
        ] = None,
        verbose: bool = True,
    ):
        self.verbose = verbose
        if self.verbose:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
            self.logger.info("Initializing ModelAbliterator")
        else:
            self.logger = None

        self.MODEL_PATH = model
        self.n_devices = n_devices or (
            torch.cuda.device_count() if torch.cuda.is_available() else 1
        )

        torch.set_grad_enabled(False)

        self.model = HookedTransformer.from_pretrained_no_processing(
            model,
            n_devices=self.n_devices,
            device=device,
            dtype=torch.bfloat16,
            default_padding_side="left",
        )

        self.model.requires_grad_(False)
        self.model.tokenizer.padding_side = "left"
        self.model.tokenizer.pad_token = self.model.tokenizer.eos_token

        self.chat_template = chat_template or ChatTemplate(self, LLAMA3_CHAT_TEMPLATE)
        self.hidden_size = self.model.cfg.d_model
        self.original_state = {
            k: v.to("cpu") for k, v in self.model.state_dict().items()
        }
        self.trait = {}
        self.baseline = {}
        self.modified_layers = {"mlp": {}, "W_O": {}}
        self.checkpoints = []

        if cache_fname:
            outs = torch.load(cache_fname, map_location="cpu")
            self.trait, self.baseline, modified_layers, checkpoints = outs[:4]
            self.checkpoints = checkpoints or []
            self.modified_layers = modified_layers

        self.trait_inst_train, self.trait_inst_test = prepare_dataset(dataset[0])
        self.baseline_inst_train, self.baseline_inst_test = prepare_dataset(dataset[1])

        self.fwd_hooks = []
        self.modified = False
        self.activation_layers = (
            activation_layers
            if isinstance(activation_layers, list)
            else [activation_layers]
        )

        self.negative_toks = negative_toks or {
            4250,
            14931,
            89735,
            20451,
            11660,
            11458,
            956,
        }
        self.positive_toks = positive_toks or {32, 1271, 8586, 96556, 78145}
        self._blacklisted = set()

    def __enter__(self):
        if hasattr(self, "current_state"):
            raise Exception("Cannot do multi-contexting")
        self.current_state = self.model.state_dict()
        self.current_layers = self.modified_layers.copy()
        self.was_modified = self.modified
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.model.load_state_dict(self.current_state)
        del self.current_state
        self.modified_layers = self.current_layers
        del self.current_layers
        self.modified = self.was_modified
        del self.was_modified

    def reset_state(self):
        self.modified = False
        self.modified_layers = {"mlp": {}, "W_O": {}}
        self.model.load_state_dict(self.original_state)

    def checkpoint(self):
        self.checkpoints.append(self.modified_layers.copy())

    def blacklist_layer(self, layer: Union[int, List[int]]):
        if isinstance(layer, list):
            for l in layer:
                self._blacklisted.add(l)
        else:
            self._blacklisted.add(layer)

    def whitelist_layer(self, layer: Union[int, List[int]]):
        if isinstance(layer, list):
            for l in layer:
                self._blacklisted.discard(l)
        else:
            self._blacklisted.discard(layer)

    def save_activations(self, fname: str):
        torch.save(
            [self.trait, self.baseline, self.modified_layers, self.checkpoints], fname
        )

    def get_whitelisted_layers(self) -> List[int]:
        return [l for l in range(self.model.cfg.n_layers) if l not in self._blacklisted]

    def get_all_act_names(
        self, activation_layers: List[str] = None
    ) -> List[Tuple[int, str]]:
        return [
            (i, utils.get_act_name(act_name, i))
            for i in self.get_whitelisted_layers()
            for act_name in (activation_layers or self.activation_layers)
        ]

    def calculate_mean_dirs(
        self, key: str, include_overall_mean: bool = False
    ) -> Dict[str, Float[Tensor, "d_model"]]:
        dirs = {
            "trait_mean": torch.mean(self.trait[key], dim=0),
            "baseline_mean": torch.mean(self.baseline[key], dim=0),
        }

        if include_overall_mean:
            if (
                self.trait[key].shape != self.baseline[key].shape
                or self.trait[key].device.type == "cuda"
            ):
                dirs["mean_dir"] = torch.mean(
                    torch.cat((self.trait[key], self.baseline[key]), dim=0), dim=0
                )
            else:
                dirs["mean_dir"] = (
                    torch.mean(self.trait[key] + self.baseline[key], dim=0) / 2.0
                )

        return dirs

    def get_avg_projections(
        self, key: str, direction: Float[Tensor, "d_model"]
    ) -> Tuple[Float[Tensor, "d_model"], Float[Tensor, "d_model"]]:
        dirs = self.calculate_mean_dirs(key)
        return torch.dot(dirs["trait_mean"], direction), torch.dot(
            dirs["baseline_mean"], direction
        )

    def get_layer_dirs(
        self, layer: int, key: str = None, include_overall_mean: bool = False
    ) -> Dict[str, Float[Tensor, "d_model"]]:
        act_key = key or self.activation_layers[0]
        if len(self.trait[key]) < layer:
            raise IndexError("Invalid layer")
        return self.calculate_mean_dirs(
            utils.get_act_name(act_key, layer),
            include_overall_mean=include_overall_mean,
        )

    def test_single_prompt(
        self, prompt: str, max_tokens_generated: int = 64, **kwargs
    ) -> str:
        # Tokenize the single prompt
        toks = self.tokenize_instructions_fn([prompt])

        # Run the model with cache
        logits, cache = self.run_with_cache(
            toks, max_new_tokens=max_tokens_generated, drop_refusals=False, **kwargs
        )

        # Decode the generated tokens
        generated_text = self.model.tokenizer.batch_decode(
            toks, skip_special_tokens=True
        )
        return generated_text[0] if generated_text else ""

    def refusal_dirs(self, invert: bool = False) -> Dict[str, Float[Tensor, "d_model"]]:
        if not self.trait:
            raise IndexError("No cache")

        refusal_dirs = {
            key: self.calculate_mean_dirs(key) for key in self.trait if ".0." not in key
        }
        if invert:
            refusal_dirs = {
                key: v["baseline_mean"] - v["trait_mean"]
                for key, v in refusal_dirs.items()
            }
        else:
            refusal_dirs = {
                key: v["trait_mean"] - v["baseline_mean"]
                for key, v in refusal_dirs.items()
            }

        return {key: (v / v.norm()).to("cpu") for key, v in refusal_dirs.items()}

    def mean_of_differences_dirs(
        self, invert: bool = False
    ) -> Dict[str, Float[Tensor, "d_model"]]:
        if not self.trait:
            raise IndexError("No cache")

        mean_of_differences = {}
        for key in self.trait:
            if ".0." in key:
                continue

            differences = self.trait[key] - self.baseline[key]
            mean_difference = torch.mean(differences, dim=0)

            if invert:
                mean_difference = -mean_difference

            mean_of_differences[key] = (mean_difference / mean_difference.norm()).to(
                "cpu"
            )

        return mean_of_differences

    def scored_dirs(
        self, invert: bool = False
    ) -> List[Tuple[str, Float[Tensor, "d_model"]]]:
        refusals = self.refusal_dirs(invert=invert)
        return sorted(
            [(ln, refusals[act_name]) for ln, act_name in self.get_all_act_names()],
            reverse=True,
            key=lambda x: abs(x[1].mean()),
        )

    def get_layer_of_act_name(self, ref: str) -> Union[str, int]:
        s = re.search(r"\.(\d+)\.", ref)
        return s if s is None else int(s[1])

    def layer_attn(
        self, layer: int, replacement: Float[Tensor, "d_model"] = None
    ) -> Float[Tensor, "d_model"]:
        if replacement is not None and layer not in self._blacklisted:
            self.modified = True
            self.model.blocks[layer].attn.W_O.data = replacement.to(
                self.model.blocks[layer].attn.W_O.device
            )
            self.modified_layers["W_O"][layer] = self.modified_layers.get(layer, []) + [
                (
                    self.model.blocks[layer].attn.W_O.data.to("cpu"),
                    replacement.to("cpu"),
                )
            ]
        return self.model.blocks[layer].attn.W_O.data

    def layer_mlp(
        self, layer: int, replacement: Float[Tensor, "d_model"] = None
    ) -> Float[Tensor, "d_model"]:
        if replacement is not None and layer not in self._blacklisted:
            self.modified = True
            self.model.blocks[layer].mlp.W_out.data = replacement.to(
                self.model.blocks[layer].mlp.W_out.device
            )
            self.modified_layers["mlp"][layer] = self.modified_layers.get(layer, []) + [
                (
                    self.model.blocks[layer].mlp.W_out.data.to("cpu"),
                    replacement.to("cpu"),
                )
            ]
        return self.model.blocks[layer].mlp.W_out.data

    def tokenize_instructions_fn(
        self, instructions: List[str]
    ) -> Int[Tensor, "batch_size seq_len"]:
        prompts = [
            self.chat_template.format(instruction=instruction)
            for instruction in instructions
        ]
        return self.model.tokenizer(
            prompts, padding=True, truncation=False, return_tensors="pt"
        ).input_ids

    def generate_logits(
        self,
        toks: Int[Tensor, "batch_size seq_len"],
        *args,
        drop_refusals: bool = True,
        stop_at_eos: bool = False,
        max_tokens_generated: int = 1,
        **kwargs,
    ) -> Tuple[
        Float[Tensor, "batch_size seq_len d_vocab"], Int[Tensor, "batch_size seq_len"]
    ]:
        all_toks = torch.zeros(
            (toks.shape[0], toks.shape[1] + max_tokens_generated),
            dtype=torch.long,
            device=toks.device,
        )
        all_toks[:, : toks.shape[1]] = toks
        generating = list(range(toks.shape[0]))
        for i in range(max_tokens_generated):
            logits = self.model(
                all_toks[generating, : -max_tokens_generated + i], *args, **kwargs
            )
            next_tokens = logits[:, -1, :].argmax(dim=-1).to("cpu")
            all_toks[generating, -max_tokens_generated + i] = next_tokens
            if drop_refusals and any(
                negative_tok in next_tokens for negative_tok in self.negative_toks
            ):
                break
            if stop_at_eos:
                generating = [
                    i
                    for i in range(toks.shape[0])
                    if all_toks[i][-1] != self.model.tokenizer.eos_token_id
                ]
                if not generating:
                    break
        return logits, all_toks

    def generate(
        self,
        prompt: Union[List[str], str],
        *model_args,
        max_tokens_generated: int = 64,
        stop_at_eos: bool = True,
        **model_kwargs,
    ) -> List[str]:
        if isinstance(prompt, str):
            gen = self.tokenize_instructions_fn([prompt])
        else:
            gen = self.tokenize_instructions_fn(prompt)

        logits, all_toks = self.generate_logits(
            gen,
            *model_args,
            stop_at_eos=stop_at_eos,
            max_tokens_generated=max_tokens_generated,
            **model_kwargs,
        )
        return self.model.tokenizer.batch_decode(all_toks, skip_special_tokens=True)

    def test(
        self,
        *args,
        test_set: List[str] = None,
        N: int = 16,
        batch_size: int = 4,
        **kwargs,
    ):
        if test_set is None:
            test_set = self.trait_inst_test
        for prompts in batch(test_set[: min(len(test_set), N)], batch_size):
            for i, res in enumerate(self.generate(prompts, *args, **kwargs)):
                if self.verbose:
                    self.logger.info(f"Result {i}: {res}")

    def run_with_cache(
        self,
        *model_args,
        names_filter: Callable[[str], bool] = None,
        incl_bwd: bool = False,
        device: str = None,
        remove_batch_dim: bool = False,
        reset_hooks_end: bool = True,
        clear_contexts: bool = False,
        fwd_hooks: List[str] = [],
        max_new_tokens: int = 1,
        **model_kwargs,
    ) -> Tuple[
        Float[Tensor, "batch_size seq_len d_vocab"],
        Dict[str, Float[Tensor, "batch_size seq_len d_model"]],
    ]:
        if names_filter is None and self.activation_layers:

            def activation_layering(namefunc: str):
                return any(s in namefunc for s in self.activation_layers)

            names_filter = activation_layering

        cache_dict, fwd, bwd = self.model.get_caching_hooks(
            names_filter,
            incl_bwd,
            device,
            remove_batch_dim=remove_batch_dim,
            pos_slice=utils.Slice(None),
        )

        fwd_hooks = fwd_hooks + fwd + self.fwd_hooks

        if not max_new_tokens:
            max_new_tokens = 1

        with self.model.hooks(
            fwd_hooks=fwd_hooks,
            bwd_hooks=bwd,
            reset_hooks_end=reset_hooks_end,
            clear_contexts=clear_contexts,
        ):
            model_out, toks = self.generate_logits(
                *model_args, max_tokens_generated=max_new_tokens, **model_kwargs
            )
            if incl_bwd:
                model_out.backward()

        return model_out, cache_dict

    def apply_refusal_dirs(
        self,
        refusal_dirs: List[Float[Tensor, "d_model"]],
        W_O: bool = True,
        mlp: bool = True,
        layers: List[str] = None,
    ):
        if layers is None:
            layers = list(range(1, self.model.cfg.n_layers))
        for refusal_dir in refusal_dirs:
            for layer in layers:
                for modifying, func in [(W_O, self.layer_attn), (mlp, self.layer_mlp)]:
                    if modifying:
                        matrix = func(layer)
                        if refusal_dir.device != matrix.device:
                            refusal_dir = refusal_dir.to(matrix.device)
                        proj = (
                            einops.einsum(
                                matrix,
                                refusal_dir.view(-1, 1),
                                "... d_model, d_model single -> ... single",
                            )
                            * refusal_dir
                        )
                        func(layer, matrix - proj)

    def induce_refusal_dir(
        self,
        refusal_dir: Float[Tensor, "d_model"],
        W_O: bool = True,
        mlp: bool = True,
        layers: List[str] = None,
    ):
        if layers is None:
            layers = list(range(1, self.model.cfg.n_layers))
        for layer in layers:
            for modifying, func in [(W_O, self.layer_attn), (mlp, self.layer_mlp)]:
                if modifying:
                    matrix = func(layer)
                    if refusal_dir.device != matrix.device:
                        refusal_dir = refusal_dir.to(matrix.device)
                    proj = (
                        einops.einsum(
                            matrix,
                            refusal_dir.view(-1, 1),
                            "... d_model, d_model single -> ... single",
                        )
                        * refusal_dir
                    )
                    avg_proj = refusal_dir * self.get_avg_projections(
                        utils.get_act_name(self.activation_layers[0], layer),
                        refusal_dir,
                    )
                    func(layer, (matrix - proj) + avg_proj)

    def test_dir(
        self,
        refusal_dir: Float[Tensor, "d_model"],
        activation_layers: List[str] = None,
        use_hooks: bool = True,
        layers: List[str] = None,
        **kwargs,
    ) -> Dict[str, Float[Tensor, "d_model"]]:
        before_hooks = self.fwd_hooks
        try:
            if layers is None:
                layers = self.get_whitelisted_layers()

            if activation_layers is None:
                activation_layers = self.activation_layers

            if use_hooks:
                hooks = self.fwd_hooks
                hook_fn = functools.partial(directional_hook, direction=refusal_dir)
                self.fwd_hooks = before_hooks + [
                    (act_name, hook_fn) for ln, act_name in self.get_all_act_names()
                ]
                return self.measure_scores(**kwargs)
            else:
                with self:
                    self.apply_refusal_dirs([refusal_dir], layers=layers)
                    return self.measure_scores(**kwargs)
        finally:
            self.fwd_hooks = before_hooks

    def find_best_refusal_dir(
        self,
        N: int = 4,
        positive: bool = False,
        use_hooks: bool = True,
        invert: bool = False,
    ) -> List[Tuple[float, str]]:
        dirs = self.refusal_dirs(invert=invert)
        if self.modified:
            print(
                "WARNING: Modified; will restore model to current modified state each run"
            )
        scores = []
        for direction in tqdm(dirs.items()):
            score = self.test_dir(direction[1], N=N, use_hooks=use_hooks)[int(positive)]
            scores.append((score, direction))
        return sorted(scores, key=lambda x: x[0])

    def measure_scores(
        self,
        N: int = 4,
        sampled_token_ct: int = 8,
        measure: str = "max",
        batch_measure: str = "max",
        positive: bool = False,
    ) -> Dict[str, Float[Tensor, "d_model"]]:
        toks = self.tokenize_instructions_fn(instructions=self.trait_inst_test[:N])
        logits, cache = self.run_with_cache(
            toks, max_new_tokens=sampled_token_ct, drop_refusals=False
        )

        negative_score, positive_score = self.measure_scores_from_logits(
            logits, sampled_token_ct, measure=batch_measure
        )

        negative_score = measure_fn(measure, negative_score)
        positive_score = measure_fn(measure, positive_score)
        return {
            "negative": negative_score.to("cpu"),
            "positive": positive_score.to("cpu"),
        }

    def measure_scores_from_logits(
        self,
        logits: Float[Tensor, "batch_size seq_len d_vocab"],
        sequence: int,
        measure: str = "max",
    ) -> Tuple[Float[Tensor, "batch_size"], Float[Tensor, "batch_size"]]:
        normalized_scores = torch.softmax(logits[:, -sequence:, :].to("cpu"), dim=-1)[
            :, :, list(self.positive_toks) + list(self.negative_toks)
        ]

        normalized_positive, normalized_negative = torch.split(
            normalized_scores, [len(self.positive_toks), len(self.negative_toks)], dim=2
        )

        max_negative_score_per_sequence = torch.max(normalized_negative, dim=-1)[0]
        max_positive_score_per_sequence = torch.max(normalized_positive, dim=-1)[0]

        negative_score_per_batch = measure_fn(
            measure, max_negative_score_per_sequence, dim=-1
        )[0]
        positive_score_per_batch = measure_fn(
            measure, max_positive_score_per_sequence, dim=-1
        )[0]
        return negative_score_per_batch, positive_score_per_batch

    def do_resid(
        self, fn_name: str
    ) -> Tuple[
        Float[Tensor, "layer batch d_model"],
        Float[Tensor, "layer batch d_model"],
        List[str],
    ]:
        if not any("resid" in k for k in self.baseline.keys()):
            raise AssertionError(
                "You need residual streams to decompose layers! Run cache_activations with None in `activation_layers`"
            )
        resid_trait, labels = getattr(self.trait, fn_name)(
            apply_ln=True, return_labels=True
        )
        resid_baseline = getattr(self.baseline, fn_name)(apply_ln=True)

        return resid_trait, resid_baseline, labels

    def decomposed_resid(
        self,
    ) -> Tuple[
        Float[Tensor, "layer batch d_model"],
        Float[Tensor, "layer batch d_model"],
        List[str],
    ]:
        return self.do_resid("decompose_resid")

    def accumulated_resid(
        self,
    ) -> Tuple[
        Float[Tensor, "layer batch d_model"],
        Float[Tensor, "layer batch d_model"],
        List[str],
    ]:
        return self.do_resid("accumulated_resid")

    def unembed_resid(
        self, resid: Float[Tensor, "layer batch d_model"], pos: int = -1
    ) -> Float[Tensor, "layer batch d_vocab"]:
        W_U = self.model.W_U
        if pos is None:
            return einops.einsum(
                resid.to(W_U.device),
                W_U,
                "layer batch d_model, d_model d_vocab -> layer batch d_vocab",
            ).to("cpu")
        else:
            return einops.einsum(
                resid[:, pos, :].to(W_U.device),
                W_U,
                "layer d_model, d_model d_vocab -> layer d_vocab",
            ).to("cpu")

    def create_layer_rankings(
        self,
        token_set: Union[List[int], Set[int], Int[Tensor, "..."]],
        decompose: bool = True,
        token_set_b: Union[List[int], Set[int], Int[Tensor, "..."]] = None,
    ) -> List[Tuple[int, int]]:
        decomposer = self.decomposed_resid if decompose else self.accumulated_resid

        decomposed_resid_trait, decomposed_resid_baseline, labels = decomposer()

        W_U = self.model.W_U.to("cpu")
        unembedded_trait = self.unembed_resid(decomposed_resid_trait)
        unembedded_baseline = self.unembed_resid(decomposed_resid_baseline)

        sorted_trait_indices = torch.argsort(unembedded_trait, dim=1, descending=True)
        sorted_baseline_indices = torch.argsort(
            unembedded_baseline, dim=1, descending=True
        )

        trait_set = torch.isin(sorted_trait_indices, torch.tensor(list(token_set)))
        baseline_set = torch.isin(
            sorted_baseline_indices,
            torch.tensor(list(token_set if token_set_b is None else token_set_b)),
        )

        indices_in_set = zip(
            trait_set.nonzero(as_tuple=True)[1], baseline_set.nonzero(as_tuple=True)[1]
        )
        return indices_in_set

    def mse_positive(
        self, N: int = 128, batch_size: int = 8, last_indices: int = 1
    ) -> Dict[str, Float[Tensor, "d_model"]]:
        toks = self.tokenize_instructions_fn(
            instructions=self.trait_inst_train[:N] + self.baseline_inst_train[:N]
        )
        splitpos = min(N, len(self.trait_inst_train))
        toks = toks[splitpos:]
        self.loss_baseline = {}

        for i in tqdm(range(0, min(N, len(toks)), batch_size)):
            logits, cache = self.run_with_cache(
                toks[i : min(i + batch_size, len(toks))]
            )
            for key in cache:
                if any(k in key for k in self.activation_layers):
                    tensor = torch.mean(cache[key][:, -last_indices:, :], dim=1).to(
                        "cpu"
                    )
                    if key not in self.loss_baseline:
                        self.loss_baseline[key] = tensor
                    else:
                        self.loss_baseline[key] = torch.cat(
                            (self.loss_baseline[key], tensor), dim=0
                        )
            del logits, cache
            clear_mem()

        return {
            k: F.mse_loss(
                self.loss_baseline[k].float()[:N], self.baseline[k].float()[:N]
            )
            for k in self.loss_baseline
        }

    def create_activation_cache(
        self,
        toks,
        N: int = 128,
        batch_size: int = 8,
        last_indices: int = 1,
        measure_refusal: int = 0,
        stop_at_layer: int = None,
    ) -> Tuple[ActivationCache, List[str]]:
        base = dict()
        z_label = [] if measure_refusal > 1 else None
        for i in tqdm(range(0, min(N, len(toks)), batch_size)):
            logits, cache = self.run_with_cache(
                toks[i : min(i + batch_size, len(toks))],
                max_new_tokens=measure_refusal,
                stop_at_layer=stop_at_layer,
            )
            if measure_refusal > 1:
                z_label.extend(
                    self.measure_scores_from_logits(logits, measure_refusal)[0]
                )
            for key in cache:
                if self.activation_layers is None or any(
                    k in key for k in self.activation_layers
                ):
                    tensor = torch.mean(
                        cache[key][:, -last_indices:, :].to("cpu"), dim=1
                    )
                    if key not in base:
                        base[key] = tensor
                    else:
                        base[key] = torch.cat((base[key], tensor), dim=0)

            del logits, cache
            clear_mem()

        return ActivationCache(base, self.model), z_label

    def cache_activations(
        self,
        N: int = 128,
        batch_size: int = 8,
        measure_refusal: int = 0,
        last_indices: int = 1,
        reset: bool = True,
        activation_layers: int = -1,
        preserve_baseline: bool = True,
        stop_at_layer: int = None,
    ):
        if hasattr(self, "current_state"):
            print("WARNING: Caching activations using a context")
        if self.modified:
            print("WARNING: Running modified model")

        if activation_layers == -1:
            activation_layers = self.activation_layers

        baseline_is_set = len(getattr(self, "baseline", {})) > 0
        preserve_baseline = baseline_is_set and preserve_baseline

        if reset or getattr(self, "baseline", None) is None:
            self.trait = {}
            if not preserve_baseline:
                self.baseline = {}

            self.trait_z_label = []
            self.baseline_z_label = []

        toks = self.tokenize_instructions_fn(
            instructions=self.trait_inst_train[:N] + self.baseline_inst_train[:N]
        )

        splitpos = min(N, len(self.trait_inst_train))
        trait_toks = toks[:splitpos]
        baseline_toks = toks[splitpos:]

        last_indices = last_indices or 1

        self.trait, self.trait_z_label = self.create_activation_cache(
            trait_toks,
            N=N,
            batch_size=batch_size,
            last_indices=last_indices,
            measure_refusal=measure_refusal,
            stop_at_layer=None,
        )
        if not preserve_baseline:
            self.baseline, self.baseline_z_label = self.create_activation_cache(
                baseline_toks,
                N=N,
                batch_size=batch_size,
                last_indices=last_indices,
                measure_refusal=measure_refusal,
                stop_at_layer=None,
            )
