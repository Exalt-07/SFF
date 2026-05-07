import json
import math
import os
import re
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from datasets import load_dataset
from transformers import AutoTokenizer, LlamaConfig
from peft import LoraConfig, get_peft_model
from safetensors.torch import save_file as st_save_file
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

from modelsplit_v2 import LlamaModel_Client, LlamaModel_Server
from utils import load_multiple_safetensors

# =============================================================================
# Arguments
# =============================================================================
parser = argparse.ArgumentParser(description="SFL-V2 training for Llama3 on GSM8K")
parser.add_argument('--cut_layer', type=int, default=8,
                    help='Layer index where the model is split between client and server')
parser.add_argument('--num_clients', type=int, default=10,
                    help='Total number of clients; dataset is split among all of them')
parser.add_argument('--client_fraction', type=float, default=0.3,
                    help='Fraction of clients selected to participate each round')
parser.add_argument('--rounds', type=int, default=10,
                    help='Number of communication rounds (each round = one aggregation step)')
parser.add_argument('--local_epochs', type=int, default=1,
                    help='Number of passes each selected client makes over its local data per round')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--lora_r', type=int, default=16,
                    help='Default LoRA rank for all clients (overridden per-client by --lora_ranks)')
parser.add_argument('--max_length', type=int, default=1024, help='Max tokenization length')
parser.add_argument('--data_fraction', type=float, default=1.0,
                    help='Fraction of GSM8K train split to use')
parser.add_argument('--dirichlet_alpha', type=float, default=0.5,
                    help='Dirichlet concentration parameter for non-IID split')
parser.add_argument('--output_dir', type=str, default='./model_v2',
                    help='Directory to save model checkpoints and LoRA adapters')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed for client sampling and data split')

# --- NEW: aggregation method ---
parser.add_argument('--agg_method', type=str, default='fedavg',
                    choices=['fedavg', 'ffa', 'flora', 'flexlora'],
                    help='Aggregation scheme to use across clients each round')

# --- NEW: heterogeneous per-client LoRA ranks ---
parser.add_argument('--lora_ranks', type=str, default=None,
                    help='Comma-separated per-client LoRA ranks, e.g. "4,8,16,8,4". '
                         'Length must equal --num_clients. If None, all clients use --lora_r.')

args = parser.parse_args()

# =============================================================================
# Derive client_ranks and validate
# =============================================================================
if args.lora_ranks:
    client_ranks = [int(r.strip()) for r in args.lora_ranks.split(',')]
    if len(client_ranks) != args.num_clients:
        raise ValueError(
            f"--lora_ranks has {len(client_ranks)} entries but --num_clients={args.num_clients}. "
            "They must match."
        )
else:
    # Homogeneous: every client uses the same rank
    client_ranks = [args.lora_r] * args.num_clients

max_lora_rank = max(client_ranks)

# FFA requires all clients to have the same rank (lora_A must be identical everywhere)
if args.agg_method == 'ffa' and len(set(client_ranks)) > 1:
    raise ValueError(
        "FFA requires all clients to share the same LoRA rank (lora_A must be identical). "
        "Pass a single --lora_r value or do not use --lora_ranks with --agg_method ffa."
    )

cut_layer        = args.cut_layer
num_clients      = args.num_clients
client_fraction  = args.client_fraction
num_rounds       = args.rounds
local_epochs     = args.local_epochs
learning_rate    = args.lr
num_participating = max(1, int(num_clients * client_fraction))

random.seed(args.seed)

print("=" * 70)
print("SFL-V2 Configuration")
print(f"  Total clients      : {num_clients}")
print(f"  Client fraction    : {client_fraction}  →  {num_participating} client(s) per round")
print(f"  Rounds             : {num_rounds}")
print(f"  Local epochs/round : {local_epochs}")
print(f"  Cut layer          : {cut_layer}")
print(f"  Aggregation method : {args.agg_method}")
print(f"  LoRA ranks         : {client_ranks[:min(10, num_clients)]}"
      + (" ..." if num_clients > 10 else ""))
print(f"  Max LoRA rank      : {max_lora_rank}")
print(f"  Dirichlet alpha    : {args.dirichlet_alpha}")
print("=" * 70)

# =============================================================================
# Environment
# =============================================================================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# =============================================================================
# Pretrained weights + tokenizer
# =============================================================================
model_name = "/projects/bewi/hramesh/SplitLoRA/models/Meta-Llama-3-8B-Instruct"

file_paths = [f"{model_name}/model-0000{i}-of-00004.safetensors" for i in range(1, 5)]
pretrained_state_dict = load_multiple_safetensors(file_paths)

configuration = LlamaConfig.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# =============================================================================
# Dataset + Dirichlet non-IID split
# =============================================================================
gsm8k_train = load_dataset("gsm8k", "main", split="train")

def format_example(example):
    messages = [
        {"role": "user", "content": (
            "Solve the following math problem step by step.\n\n"
            f"Question: {example['question']}"
        )},
        {"role": "assistant", "content": example['answer']},
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

dataset = gsm8k_train.map(format_example)
dataset = dataset.select(range(int(args.data_fraction * len(dataset))))

def _gsm8k_label(example):
    match = re.search(r'####\s*([\d,]+)', example['answer'])
    if match:
        return int(match.group(1).replace(',', '')) % 10
    return 0

def dirichlet_noniid_split(dataset, num_clients, alpha, seed):
    rng = np.random.RandomState(seed)
    labels = np.array([_gsm8k_label(dataset[i]) for i in range(len(dataset))])
    num_classes = 10
    client_indices = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        class_idxs = np.where(labels == c)[0]
        if len(class_idxs) == 0:
            continue
        rng.shuffle(class_idxs)
        proportions = rng.dirichlet(alpha * np.ones(num_clients))
        splits = (np.cumsum(proportions[:-1]) * len(class_idxs)).astype(int)
        for client_i, chunk in enumerate(np.split(class_idxs, splits)):
            client_indices[client_i].extend(chunk.tolist())
    for i in range(num_clients):
        if len(client_indices[i]) == 0:
            donor = max(range(num_clients), key=lambda j: len(client_indices[j]))
            client_indices[i].append(client_indices[donor].pop())
    subsets = [torch.utils.data.Subset(dataset, sorted(idxs)) for idxs in client_indices]
    sizes   = [len(s) for s in subsets]
    return subsets, sizes

total_size = len(dataset)
client_datasets, client_sizes = dirichlet_noniid_split(
    dataset, num_clients, alpha=args.dirichlet_alpha, seed=args.seed
)
train_dataloaders = [
    DataLoader(client_datasets[i], batch_size=1, shuffle=True)
    for i in range(num_clients)
]

print(f"\nDataset: {total_size} examples → Dirichlet non-IID "
      f"(alpha={args.dirichlet_alpha}) across {num_clients} clients")
print(f"  Shard sizes : {client_sizes[:min(10, num_clients)]}"
      + (" ..." if num_clients > 10 else ""))
print(f"  Min / Max   : {min(client_sizes)} / {max(client_sizes)}\n")

# =============================================================================
# Global PEFT config (used for server and for saving the global adapter).
# Always uses max_lora_rank so the saved adapter is compatible with evaluation.
# Per-client configs are created inside make_client().
# =============================================================================
global_peft_config = LoraConfig(
    r=max_lora_rank,
    lora_alpha=max_lora_rank,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj"],
)

# =============================================================================
# Debug utilities
# =============================================================================

def debug_lora_tensor(tag, key, tensor):
    """
    Print shape, norm, and health status of a single LoRA tensor.
    Flags: all-zeros (dead adapter), NaN, Inf.
    """
    if tensor is None:
        print(f"  [{tag}] {key} | NONE")
        return
    norm   = tensor.float().abs().sum().item()
    status = "OK"
    if torch.isnan(tensor).any():
        status = "CRITICAL: NaN detected"
    elif torch.isinf(tensor).any():
        status = "CRITICAL: Inf detected"
    elif norm == 0.0:
        status = "WARNING: ALL ZEROS — adapter is dead"
    print(f"  [{tag}] {key}")
    print(f"          shape={tuple(tensor.shape)}  |  norm={norm:.4f}  |  {status}")
    print(f"          {'-'*54}")


def debug_lora_state(tag, lora_state, max_layers=1):
    """
    Print debug info for the first max_layers transformer layers in a lora state dict.
    Shows one lora_A and one lora_B per projection per layer.
    """
    printed = set()
    for key, val in sorted(lora_state.items()):
        # Only show the first max_layers layers
        m = re.search(r'layers\.(\d+)', key)
        if m and int(m.group(1)) >= max_layers:
            continue
        proj = 'q_proj' if 'q_proj' in key else ('k_proj' if 'k_proj' in key else None)
        if proj is None:
            continue
        label = f"{proj}_{'A' if 'lora_A' in key else 'B'}"
        if label in printed:
            continue
        printed.add(label)
        debug_lora_tensor(tag, key, val)


def debug_flora_base_sd(tag, flora_base_sd, max_layers=1):
    """
    Print norms of client base weights after FLoRA merge.
    Used to verify delta was applied correctly.
    """
    print(f"  [{tag}] flora_base_sd — first {max_layers} layer(s):")
    for key, val in sorted(flora_base_sd.items()):
        m = re.search(r'layers\.(\d+)', key)
        if m and int(m.group(1)) >= max_layers:
            continue
        if 'q_proj' in key or 'k_proj' in key:
            norm = val.float().abs().sum().item()
            print(f"          {key}  |  norm={norm:.4f}")
    print(f"          {'-'*54}")

# =============================================================================
# LoRA pad / truncate helpers (Llama — no MergedLinear, no chunk splitting)
#
# lora_A shape : [r, in_features]   — pad/truncate rows
# lora_B shape : [out_features, r]  — pad/truncate cols
# =============================================================================

def pad_lora_A(tensor, target_rank):
    gap = target_rank - tensor.shape[0]
    if gap <= 0:
        return tensor
    return F.pad(tensor, (0, 0, 0, gap))   # pad bottom rows with zeros

def pad_lora_B(tensor, target_rank):
    gap = target_rank - tensor.shape[1]
    if gap <= 0:
        return tensor
    return F.pad(tensor, (0, gap, 0, 0))   # pad right cols with zeros

def truncate_lora_A(tensor, target_rank):
    return tensor[:target_rank, :]

def truncate_lora_B(tensor, target_rank):
    return tensor[:, :target_rank]

def _pair_key(key):
    """
    Given a lora_A key return its lora_B counterpart and vice-versa.
    Uses the full suffix so it works for any adapter name (e.g. 'default').
    Example:
      base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight
      →
      base_model.model.layers.0.self_attn.q_proj.lora_B.default.weight
    """
    if 'lora_A' in key:
        return key.replace('lora_A', 'lora_B', 1)
    return key.replace('lora_B', 'lora_A', 1)

# =============================================================================
# Aggregation helpers
# =============================================================================

# ── 1. FedAvg (and FFA — same aggregation, different freeze logic) ───────────

def fed_avg_lora_only(client_states, client_ranks_list, data_sizes=None):
    """
    Weighted FedAvg over LoRA parameters.

    Steps:
      1. Pad every client's lora_A (rows) and lora_B (cols) to max_rank.
      2. Compute weighted average → global state at max_rank.
      3. Truncate global state back to each client's own rank → per_client list.

    Returns:
      global_state      : dict  — averaged LoRA at max_rank (for eval / saving)
      per_client_states : list  — one dict per client, truncated to their rank
    """
    max_rank = max(client_ranks_list)
    n        = len(client_states)

    if data_sizes is None:
        weights = [1.0 / n] * n
    else:
        total   = sum(data_sizes)
        weights = [s / total for s in data_sizes]

    # Step 1: pad all clients to max_rank
    padded = []
    for state, rank in zip(client_states, client_ranks_list):
        p = {}
        for k, v in state.items():
            vf = v.float()
            if 'lora_A' in k:
                p[k] = pad_lora_A(vf, max_rank)
            elif 'lora_B' in k:
                p[k] = pad_lora_B(vf, max_rank)
        padded.append(p)

    # Step 2: weighted average
    lora_keys    = list(padded[0].keys())
    global_state = {}
    for key in lora_keys:
        global_state[key] = sum(
            padded[i][key] * weights[i] for i in range(n)
        ).to(client_states[0][key].dtype)

    # Step 3: truncate global → per client
    per_client_states = []
    for rank in client_ranks_list:
        pc = {}
        for k, v in global_state.items():
            if 'lora_A' in k:
                pc[k] = truncate_lora_A(v, rank)
            elif 'lora_B' in k:
                pc[k] = truncate_lora_B(v, rank)
        per_client_states.append(pc)

    return global_state, per_client_states


# ── 2. FFA — Frozen-A init/freeze ────────────────────────────────────────────

def init_and_freeze_lora_A(peft_model, seed=42):
    """
    Re-initialise every lora_A weight with a dedicated fixed-seed Generator
    (independent of the global RNG) and freeze it (requires_grad=False).

    Must be called AFTER any load_state_dict() so the freeze survives the load.
    Using a dedicated Generator ensures all client instances get bit-identical
    lora_A regardless of what other RNG-consuming operations happened between calls.
    """
    gen = torch.Generator(device='cpu')
    gen.manual_seed(seed)
    frozen_count = 0
    for name, param in peft_model.named_parameters():
        if 'lora_A' in name:
            torch.nn.init.kaiming_uniform_(param, a=math.sqrt(5), generator=gen)
            param.requires_grad_(False)
            frozen_count += 1
    print(f"  [FFA] Froze {frozen_count} lora_A parameter tensors (seed={seed})")


# ── 3. FLoRA — Stack + Merge-and-Reset ───────────────────────────────────────

def flora_stack(client_states, client_ranks_list, lora_alpha, data_sizes=None):
    """
    Stack client LoRA matrices for FLoRA.

    For each key:
      lora_A: cat(dim=0)  →  [sum(r_i), in_features]    (scale baked in)
      lora_B: cat(dim=1)  →  [out_features, sum(r_i)]   (unscaled)

    Scale baked into A: scale_i = (lora_alpha / r_i) * weight_i
    so that B_stacked @ A_stacked = Σ_i weight_i * (B_i @ A_i) * (alpha/r_i)
    which is the weighted-averaged LoRA output with standard LoRA scaling.
    The merge step uses scale=1.0 — do NOT apply lora_alpha/rank again there.

    Works naturally with heterogeneous ranks because cat handles variable sizes.
    """
    n     = len(client_states)
    total = sum(data_sizes) if data_sizes else n
    w     = [s / total for s in data_sizes] if data_sizes else [1.0 / n] * n

    a_keys  = [k for k in client_states[0] if 'lora_A' in k]
    stacked = {}

    for a_key in a_keys:
        b_key  = _pair_key(a_key)
        A_list, B_list = [], []
        for i, (state, rank) in enumerate(zip(client_states, client_ranks_list)):
            scale = (lora_alpha / rank) * w[i]
            A_list.append(state[a_key].float() * scale)   # scale baked into A
            B_list.append(state[b_key].float())            # B left unscaled
        stacked[a_key] = torch.cat(A_list, dim=0)          # [Σr_i,  in]
        stacked[b_key] = torch.cat(B_list, dim=1)          # [out,   Σr_i]

    return stacked


def flora_merge_into_sd(flora_base_sd, stacked_state):
    """
    Apply the FLoRA delta (B_stacked @ A_stacked) to flora_base_sd.

    Key mapping (PEFT → pretrained_state_dict format):
      base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight
      →  model.layers.0.self_attn.q_proj.weight

    Returns a new dict — does NOT mutate flora_base_sd in place so the
    previous round's weights are still accessible for debugging.
    """
    new_sd = {k: v.clone() for k, v in flora_base_sd.items()}

    merged_count  = 0
    missing_count = 0

    for a_key, A in stacked_state.items():
        if 'lora_A' not in a_key:
            continue
        b_key  = _pair_key(a_key)
        B      = stacked_state[b_key]
        delta  = B.float() @ A.float()   # [out, in]; scale=1.0 already in A

        # Map PEFT key → pretrained_state_dict key
        # e.g. base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight
        #    → model.layers.0.self_attn.q_proj.weight
        base_key = re.sub(
            r'\.lora_A\.[^.]+\.weight$', '.weight',
            a_key.replace('base_model.', '', 1)
        )

        if base_key in new_sd:
            new_sd[base_key] = (new_sd[base_key].float() + delta).to(torch.bfloat16)
            merged_count += 1
        else:
            print(f"  [FLoRA] WARNING: base key not found in flora_base_sd: {base_key}")
            missing_count += 1

    print(f"  [FLoRA] Merged {merged_count} weight tensors "
          f"({missing_count} missing keys — see warnings above)")
    return new_sd


# ── 4. Flex-LoRA — SVD-based re-decomposition ────────────────────────────────

def flexlora_aggregate(client_states, client_ranks_list, target_rank, data_sizes=None):
    """
    Flex-LoRA aggregation — all heavy math on GPU in FP32.

    Cast BF16 → FP32 on GPU for SVD precision (BF16 has only 7 mantissa bits,
    insufficient for accurate singular vectors). This matches GPT2's approach
    of running SVD on-device — just with an explicit FP32 cast since we train
    in BF16 rather than FP16.

    client_states are CPU tensors (stored that way to free VRAM during training).
    We move to CUDA only for the aggregation math, then cast results back to BF16.
    """
    n     = len(client_states)
    total = sum(data_sizes) if data_sizes else n
    w     = [s / total for s in data_sizes] if data_sizes else [1.0 / n] * n

    a_keys       = [k for k in client_states[0] if 'lora_A' in k]
    global_state = {}

    for a_key in a_keys:
        b_key = _pair_key(a_key)

        # Step 1+2: weighted sum of B_k @ A_k
        # .cuda().float() : move CPU BF16 tensor → GPU FP32 in one step
        delta_avg = sum(
            (client_states[i][b_key].cuda().float()
             @ client_states[i][a_key].cuda().float()) * w[i]
            for i in range(n)
        )   # [out_features, in_features], FP32 on GPU

        # Step 3: Economy SVD — GPU FP32, fast (~10-50ms vs ~1-3s on CPU)
        U, S, Vh = torch.linalg.svd(delta_avg, full_matrices=False)
        # U  : [out, k]   S  : [k]   Vh : [k, in]   where k = min(out, in)

        r = min(target_rank, S.shape[0])

        # Step 4: Reconstruct at target_rank, cast back to BF16
        ref_dtype = client_states[0][a_key].dtype   # original BF16
        A_new = Vh[:r, :].to(ref_dtype)                        # [r, in]
        B_new = (U[:, :r] @ torch.diag(S[:r])).to(ref_dtype)   # [out, r]

        # Store back on CPU to match how other methods store their states
        global_state[a_key] = A_new.cpu()
        global_state[b_key] = B_new.cpu()

        # Debug: singular value energy capture
        energy_r   = S[:r].sum().item()
        energy_all = S.sum().item()
        layer_name = re.search(r'layers\.\d+\.[^.]+(?:q_proj|k_proj)', a_key)
        layer_name = layer_name.group(0) if layer_name else a_key
        print(f"  [FlexLoRA] {layer_name} | "
              f"top-{r} SV energy={100*energy_r/max(energy_all,1e-9):.1f}% | "
              f"S[0]={S[0].item():.4f}  S[{r-1}]={S[r-1].item():.6f}")

        # Free GPU memory immediately after each layer
        del delta_avg, U, S, Vh, A_new, B_new
        torch.cuda.empty_cache()

    # Truncate global → per client
    per_client_states = []
    for rank in client_ranks_list:
        pc = {}
        for k, v in global_state.items():
            if 'lora_A' in k:
                pc[k] = truncate_lora_A(v, rank)
            elif 'lora_B' in k:
                pc[k] = truncate_lora_B(v, rank)
        per_client_states.append(pc)

    return global_state, per_client_states
# =============================================================================
# Model helpers
# =============================================================================

def load_pretrain_split_v2(client_model, server_model, lm_head, pretrained_sd, cut_layer):
    """Load pretrained weights, remapping server layer indices."""
    state_dict = {k.replace('model.', ''): v for k, v in pretrained_sd.items()}
    client_update, server_update = {}, {}
    for key, value in state_dict.items():
        if 'layers.' in key:
            layer_num = int(key.split('.')[1])
            if layer_num < cut_layer:
                client_update[key] = value
            else:
                new_key = key.replace(
                    f'layers.{layer_num}', f'layers.{layer_num - cut_layer}'
                )
                server_update[new_key] = value
        elif key == 'embed_tokens.weight':
            client_update[key] = value
        elif key == 'norm.weight':
            server_update[key] = value
    lm_head.weight.data = pretrained_sd['lm_head.weight'].to(torch.float32)
    client_model.load_state_dict(client_update, strict=False)
    server_model.load_state_dict(server_update, strict=False)


def make_client(rank, agg_method='fedavg',
                client_lora_state=None,
                flora_base_sd=None):
    """
    Build a fresh client PEFT model.

    rank             : LoRA rank for this specific client.
    agg_method       : Controls whether lora_A is frozen (ffa) and whether
                       flora_base_sd is applied to base weights (flora).
    client_lora_state: Per-client truncated LoRA state from last aggregation.
                       None on the first round.
    flora_base_sd    : Accumulated merged base weights for FLoRA.
                       None for all other methods.
    """
    # Per-client LoRA config at this client's rank
    peft_config_i = LoraConfig(
        r=rank,
        lora_alpha=rank,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj"],
    )

    base = LlamaModel_Client(configuration, cut_layer)

    # Build client-side pretrained weight dict (strips 'model.' prefix)
    state_dict   = {k.replace('model.', ''): v for k, v in pretrained_state_dict.items()}
    client_update = {}
    for key, value in state_dict.items():
        if 'layers.' in key:
            if int(key.split('.')[1]) < cut_layer:
                client_update[key] = value
        elif key == 'embed_tokens.weight':
            client_update[key] = value

    # FLoRA: override base weights with the post-merge accumulated weights.
    # flora_base_sd keys use 'model.' prefix → strip to match client_update format.
    if flora_base_sd is not None:
        overridden = 0
        for fkey, fval in flora_base_sd.items():
            bare = fkey.replace('model.', '', 1)
            if bare in client_update:
                client_update[bare] = fval.clone()
                overridden += 1
        print(f"  [make_client] FLoRA: overrode {overridden} base weight tensors "
              f"with merged weights (rank={rank})")

    base.load_state_dict(client_update, strict=False)
    base   = base.to(torch.bfloat16)
    client = get_peft_model(base, peft_config_i)

    # Load this client's rank-truncated LoRA state from the previous round.
    # Skip for FLoRA — clients get a fresh LoRA init each round (the delta
    # has already been absorbed into the base weights via flora_merge_into_sd).
    if client_lora_state is not None and agg_method != 'flora':
        s = client.state_dict()
        s.update(client_lora_state)
        client.load_state_dict(s)
        print(f"  [make_client] Loaded per-client LoRA state "
              f"(rank={rank}, {len(client_lora_state)} tensors)")

    # FFA: re-freeze lora_A AFTER any load_state_dict.
    # Must come last — load_state_dict resets requires_grad to True.
    if agg_method == 'ffa':
        init_and_freeze_lora_A(client, seed=42)

    return client


def save_lora_adapter(lora_state, save_dir, peft_cfg):
    """Write a PEFT-compatible LoRA adapter folder (safetensors + adapter_config.json)."""
    os.makedirs(save_dir, exist_ok=True)
    st_save_file(
        {k: v.to(torch.bfloat16).contiguous() for k, v in lora_state.items()},
        os.path.join(save_dir, "adapter_model.safetensors"),
    )
    config_dict = peft_cfg.to_dict()
    if 'peft_type' not in config_dict:
        config_dict['peft_type'] = 'LORA'
    config_dict = {k: sorted(v) if isinstance(v, set) else v
                   for k, v in config_dict.items()}
    with open(os.path.join(save_dir, "adapter_config.json"), 'w') as f:
        json.dump(config_dict, f, indent=2)

# =============================================================================
# Model Initialization
# =============================================================================
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)
global_client_lora_path = os.path.join(output_dir, f"global_lora_client_{cut_layer}")

# Single shared server — BF16 to halve memory; no GradScaler needed because
# BF16 has the same exponent range as FP32 (no overflow risk).
server_base = LlamaModel_Server(configuration, cut_layer)
lm_head     = nn.Linear(configuration.hidden_size, configuration.vocab_size, bias=False)
_dc         = LlamaModel_Client(configuration, cut_layer)
load_pretrain_split_v2(_dc, server_base, lm_head, pretrained_state_dict, cut_layer)
del _dc
server_base = server_base.to(torch.bfloat16)
lm_head     = lm_head.to(torch.bfloat16)
server      = get_peft_model(server_base, global_peft_config)
server.to('cuda')
lm_head.to('cuda')

server_optimizer = torch.optim.AdamW(
    list(server.parameters()) + list(lm_head.parameters()),
    lr=learning_rate,
)

print(f"Shared server initialized on GPU  (lora_r={max_lora_rank})\n")

# =============================================================================
# Pre-round state
# =============================================================================

# global_lora_state : LoRA dict at max_lora_rank used for eval/save.
#                     None until after the first round's aggregation.
global_lora_state = None

# per_client_lora_states : maps client_idx → rank-truncated LoRA state for
#                          that client's next training round.
#                          Empty until after the first round's aggregation.
per_client_lora_states = {}

# flora_base_sd : CPU BF16 copy of client-side pretrained weights.
#                 After each FLoRA round the aggregated delta is folded in here.
#                 make_client() reads from here instead of pretrained_state_dict.
flora_base_sd = None
if args.agg_method == 'flora':
    flora_base_sd = {}
    for key, val in pretrained_state_dict.items():
        m = re.search(r'model\.layers\.(\d+)', key)
        if m and int(m.group(1)) < cut_layer:
            flora_base_sd[key] = val.clone().to(torch.bfloat16).cpu()
        elif 'model.embed_tokens.weight' in key:
            flora_base_sd[key] = val.clone().to(torch.bfloat16).cpu()
    print(f"[FLoRA] Initialized flora_base_sd: {len(flora_base_sd)} weight tensors\n")

# One-time flag to print PEFT key format from first real client
_peft_keys_printed = False

# =============================================================================
# Communication Rounds
# =============================================================================
for round_idx in range(num_rounds):

    # ── Client selection ──────────────────────────────────────────────────────
    participating = sorted(random.sample(range(num_clients), num_participating))
    print(f"\n{'='*70}")
    print(f"Round {round_idx+1}/{num_rounds}  |  "
          f"Selected clients: {participating}  "
          f"({num_participating}/{num_clients})  |  "
          f"Method: {args.agg_method}")
    print(f"{'='*70}")

    client_lora_states   = []
    participating_ranks  = [client_ranks[i] for i in participating]
    round_loss           = 0.0
    round_batches        = 0

    server.train()
    lm_head.train()

    # ── Per-client local training ─────────────────────────────────────────────
    for client_idx in participating:
        rank             = client_ranks[client_idx]
        client_lora_state = per_client_lora_states.get(client_idx, None)

        print(f"\n  -- Client {client_idx}  (rank={rank}) --")

        client = make_client(
            rank=rank,
            agg_method=args.agg_method,
            client_lora_state=client_lora_state,
            flora_base_sd=flora_base_sd,
        )

        # One-time: print actual PEFT key format so we can verify string ops
        #global _peft_keys_printed
        if not _peft_keys_printed:
            sample_keys = [k for k in client.state_dict() if 'lora_' in k][:6]
            print(f"\n  [DEBUG] PEFT lora key format (first 6 keys):")
            for sk in sample_keys:
                print(f"    {sk}")
            print()
            _peft_keys_printed = True

        client.to('cuda').train()
        client_optimizer = torch.optim.AdamW(client.parameters(), lr=learning_rate)

        # ── Local training: local_epochs passes over this client's shard ──
        for local_epoch in range(local_epochs):
            epoch_desc = (
                f"  Round {round_idx+1} | Client {client_idx} | "
                f"Local epoch {local_epoch+1}/{local_epochs}"
            )
            for batch in tqdm(train_dataloaders[client_idx],
                               desc=epoch_desc, leave=False):
                inputs = tokenizer(
                    batch['text'],
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=args.max_length,
                ).to('cuda')

                server_optimizer.zero_grad()
                client_optimizer.zero_grad()

                # Step 1: client forward → smashed data
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    hidden_states, causal_mask, position_ids = client(**inputs)

                # Detach: simulates smashed-data transmission
                client_hidden = hidden_states.detach().requires_grad_(True)

                # Step 2: server forward + loss
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    server_out = server(
                        hidden_states=client_hidden,
                        causal_mask=causal_mask,
                        position_ids=position_ids,
                    )
                    logits    = lm_head(server_out[0])
                    labels    = inputs['input_ids'][:, 1:].contiguous()
                    logits_s  = logits[:, :-1, :].contiguous()
                    loss = nn.CrossEntropyLoss(
                        ignore_index=tokenizer.pad_token_id
                    )(logits_s.view(-1, logits_s.size(-1)), labels.view(-1))

                # Step 3: server backward → extract activation gradient
                loss.backward()

                if client_hidden.grad is not None:
                    dfx = client_hidden.grad.clone().detach()
                    if torch.isnan(dfx).any() or torch.isinf(dfx).any():
                        print(f"  [WARNING] Bad gradient from server at "
                              f"round {round_idx+1} client {client_idx} — skipping client update")
                        dfx = None
                else:
                    dfx = None

                torch.nn.utils.clip_grad_norm_(
                    list(server.parameters()) + list(lm_head.parameters()), 1.0
                )
                server_optimizer.step()
                server_optimizer.zero_grad()

                # Step 4: client backward using server gradient signal
                if dfx is not None:
                    hidden_states.backward(dfx)
                    torch.nn.utils.clip_grad_norm_(client.parameters(), 1.0)
                    client_optimizer.step()
                client_optimizer.zero_grad()

                round_loss   += loss.item()
                round_batches += 1
                torch.cuda.empty_cache()

        print(f"  Client {client_idx} finished  "
              f"({len(client_datasets[client_idx])} examples, "
              f"{local_epochs} local epoch(s), rank={rank})")

        # Collect only LoRA keys (~ rank * d * 2 per layer)
        raw_lora = {k: v.cpu().clone()
                    for k, v in client.state_dict().items()
                    if 'lora_' in k}
        client_lora_states.append(raw_lora)

        # ── Debug: inspect raw client LoRA before aggregation ──
        print(f"\n  [DEBUG PRE-AGG] Client {client_idx} raw LoRA (layer 0):")
        debug_lora_state(f"C{client_idx}_RAW_r{rank}", raw_lora, max_layers=1)

        client.to('cpu')
        del client, client_optimizer
        torch.cuda.empty_cache()

    # ── Aggregation dispatch ──────────────────────────────────────────────────
    participating_sizes = [len(client_datasets[i]) for i in participating]

    print(f"\n  [AGG] Running '{args.agg_method}' aggregation over "
          f"{len(participating)} clients  |  ranks={participating_ranks}")

    if args.agg_method in ('fedavg', 'ffa'):
        # ── FedAvg / FFA ──────────────────────────────────────────────────
        # FFA aggregation is mathematically identical to FedAvg.
        # The freeze happens inside make_client() at the start of the next round.
        global_lora_state, per_client_results = fed_avg_lora_only(
            client_lora_states, participating_ranks, participating_sizes
        )
        # Store per-client states for next round's make_client()
        for i, cidx in enumerate(participating):
            per_client_lora_states[cidx] = per_client_results[i]

        # Debug: inspect global averaged state
        print(f"\n  [DEBUG POST-AGG] Global averaged LoRA (layer 0, max_rank={max_lora_rank}):")
        debug_lora_state(f"GLOBAL_{args.agg_method.upper()}", global_lora_state, max_layers=1)

        # Debug: inspect one per-client truncated state
        example_idx = participating[0]
        example_rank = client_ranks[example_idx]
        print(f"  [DEBUG POST-AGG] Per-client LoRA for client {example_idx} "
              f"(rank={example_rank}, layer 0):")
        debug_lora_state(
            f"C{example_idx}_TRUNCATED_r{example_rank}",
            per_client_lora_states[example_idx],
            max_layers=1
        )

    elif args.agg_method == 'flora':
        # ── FLoRA ─────────────────────────────────────────────────────────
        # Step A: Stack client LoRA matrices
        print(f"  [FLoRA] Stacking {len(client_lora_states)} client adapters ...")
        stacked = flora_stack(
            client_lora_states,
            participating_ranks,
            lora_alpha=args.lora_r,
            data_sizes=participating_sizes,
        )
        stacked_rank = sum(participating_ranks)
        print(f"  [FLoRA] Stacked rank = {stacked_rank}")

        # Debug: inspect stacked tensors
        print(f"  [DEBUG PRE-MERGE] Stacked LoRA (layer 0):")
        debug_lora_state("STACKED", stacked, max_layers=1)

        # Debug: base weights BEFORE merge
        print(f"  [DEBUG PRE-MERGE] flora_base_sd base weights (layer 0):")
        debug_flora_base_sd("BASE_PRE_MERGE", flora_base_sd, max_layers=1)

        # Step B: Merge delta into flora_base_sd and update
        old_flora_base_sd = flora_base_sd
        flora_base_sd = flora_merge_into_sd(flora_base_sd, stacked)

        # Debug: base weights AFTER merge — delta should be visible in norms
        print(f"  [DEBUG POST-MERGE] flora_base_sd base weights (layer 0):")
        debug_flora_base_sd("BASE_POST_MERGE", flora_base_sd, max_layers=1)

        # Sanity check: norms should change after merge
        for key in flora_base_sd:
            if 'q_proj' in key and 'layers.0' in key:
                old_norm = old_flora_base_sd[key].float().abs().sum().item()
                new_norm = flora_base_sd[key].float().abs().sum().item()
                print(f"  [FLoRA] Norm delta check | {key.split('layers.')[-1]} | "
                      f"before={old_norm:.4f}  after={new_norm:.4f}  "
                      f"diff={abs(new_norm-old_norm):.6f}"
                      + ("  ← ZERO CHANGE (possible bug!)"
                         if abs(new_norm - old_norm) < 1e-6 else ""))
                break

        # FLoRA clients start fresh from merged base weights — no LoRA state to carry
        global_lora_state  = None
        per_client_lora_states = {}

    elif args.agg_method == 'flexlora':
        # ── Flex-LoRA ─────────────────────────────────────────────────────
        print(f"  [FlexLoRA] SVD aggregation (target_rank={max_lora_rank}) ...")
        global_lora_state, per_client_results = flexlora_aggregate(
            client_lora_states,
            participating_ranks,
            target_rank=max_lora_rank,
            data_sizes=participating_sizes,
        )
        for i, cidx in enumerate(participating):
            per_client_lora_states[cidx] = per_client_results[i]

        # Debug: inspect SVD-reconstructed global state
        print(f"\n  [DEBUG POST-AGG] FlexLoRA global state (layer 0, rank={max_lora_rank}):")
        debug_lora_state("GLOBAL_FLEXLORA", global_lora_state, max_layers=1)

    # ── Save ──────────────────────────────────────────────────────────────────
    if global_lora_state is not None:
        # FedAvg / FFA / FlexLoRA: save global LoRA adapter at max_rank
        save_lora_adapter(global_lora_state, global_client_lora_path, global_peft_config)
        print(f"  Saved global LoRA adapter → {global_client_lora_path}")
    elif args.agg_method == 'flora' and flora_base_sd is not None:
        # FLoRA: no LoRA adapter to save — the knowledge lives in flora_base_sd
        flora_save_path = os.path.join(
            output_dir, f"flora_base_client_{cut_layer}.safetensors"
        )
        st_save_file(
            {k: v.contiguous() for k, v in flora_base_sd.items()},
            flora_save_path,
        )
        print(f"  [FLoRA] Saved merged base weights → {flora_save_path}")

    avg_loss = round_loss / max(round_batches, 1)
    print(f"\nRound {round_idx+1}/{num_rounds} complete  |  "
          f"Avg loss: {avg_loss:.4f}  |  Batches: {round_batches}")

# =============================================================================
# Save Final Models
# =============================================================================
print("\n" + "="*70)
print("Saving final models...")

# Client-side: global LoRA or FLoRA base weights
if global_lora_state is not None:
    print(f"  Global client LoRA (max_rank={max_lora_rank}) → {global_client_lora_path}")
elif args.agg_method == 'flora' and flora_base_sd is not None:
    flora_final_path = os.path.join(
        output_dir, f"flora_base_client_{cut_layer}_final.safetensors"
    )
    st_save_file(
        {k: v.contiguous() for k, v in flora_base_sd.items()},
        flora_final_path,
    )
    print(f"  [FLoRA] Final merged base weights → {flora_final_path}")

# Server LoRA
server_save_path = os.path.join(output_dir, f"lora_server_{cut_layer}")
server.save_pretrained(server_save_path)
print(f"  Server LoRA → {server_save_path}")

# LM head
lm_head_save_path = os.path.join(output_dir, f"lm_head_{cut_layer}.pt")
torch.save(lm_head.state_dict(), lm_head_save_path)
print(f"  LM head → {lm_head_save_path}")

print("\nTraining complete.")
