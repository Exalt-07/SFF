import os
import re
import torch
from torch import nn
from datasets import load_dataset
from transformers import AutoTokenizer, LlamaConfig
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file
from tqdm import tqdm
import argparse

# =============================================================================
# Arguments
# =============================================================================
parser = argparse.ArgumentParser(description="SFL-V2 GSM8K evaluation")
parser.add_argument('--cut_layer', type=int, default=8,
                    help='Cut layer used during training')
parser.add_argument('--output_dir', type=str, default='./model_v2',
                    help='Directory where the training script saved its outputs')
parser.add_argument('--client_idx', type=int, default=None,
                    help='Evaluate a specific client LoRA instead of the global averaged one '
                         '(not applicable for FLoRA)')
parser.add_argument('--max_new_tokens', type=int, default=256,
                    help='Max tokens to generate per example (256 is sufficient for GSM8K)')
parser.add_argument('--test_fraction', type=float, default=1.0,
                    help='Fraction of GSM8K test set to use (default: full 1319-example set)')
parser.add_argument('--no_lora', action='store_true',
                    help='Evaluate the pretrained base model without any adapters '
                         '(useful to verify eval correctness and establish a baseline)')

# --- NEW: must match what was passed to demo_train_SFLv2.py ---
parser.add_argument('--agg_method', type=str, default='fedavg',
                    choices=['fedavg', 'ffa', 'flora', 'flexlora'],
                    help='Aggregation method used during training. '
                         'Default fedavg — behaviour identical to original script.')
parser.add_argument('--lora_r', type=int, default=16,
                    help='LoRA rank used during training. Must equal max(--lora_ranks) '
                         'if heterogeneous ranks were used. Default 16 — matches original.')

args = parser.parse_args()

cut_layer  = args.cut_layer
output_dir = args.output_dir

# =============================================================================
# Validate: --client_idx is not meaningful for FLoRA
# =============================================================================
if args.agg_method == 'flora' and args.client_idx is not None:
    raise ValueError(
        "--client_idx is not supported for --agg_method flora. "
        "FLoRA merges all client knowledge into the base weights; there is no "
        "per-client LoRA adapter file."
    )

# =============================================================================
# Environment
# =============================================================================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# =============================================================================
# Paths — different for FLoRA vs everything else
# =============================================================================
model_name = "/projects/bewi/hramesh/SplitLoRA/models/Meta-Llama-3-8B-Instruct"

# Server LoRA and lm_head paths are identical for ALL methods
server_lora_path = os.path.join(output_dir, f"lora_server_{cut_layer}")
lm_head_path     = os.path.join(output_dir, f"lm_head_{cut_layer}.pt")

if not args.no_lora:
    if args.agg_method == 'flora':
        # FLoRA: client knowledge is in the merged base weights, not a LoRA adapter
        flora_base_path = os.path.join(
            output_dir, f"flora_base_client_{cut_layer}_final.safetensors"
        )
        # Fallback: if final save wasn't reached (e.g. job timed out), use per-round save
        if not os.path.exists(flora_base_path):
            flora_base_path = os.path.join(
                output_dir, f"flora_base_client_{cut_layer}.safetensors"
            )
            print(f"[FLoRA] NOTE: _final.safetensors not found, "
                  f"falling back to per-round save: {flora_base_path}")

        for path in [flora_base_path, server_lora_path, lm_head_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Could not find '{path}'. "
                    f"Run demo_train_SFLv2.py first with "
                    f"--cut_layer {cut_layer} --agg_method flora, "
                    f"or pass --no_lora to evaluate the base model."
                )
        print(f"[FLoRA] Client base weights : {flora_base_path}")

    else:
        # FedAvg / FFA / FlexLoRA: client knowledge is in a PEFT adapter folder
        if args.client_idx is not None:
            client_lora_path = os.path.join(
                output_dir, f"lora_client_{args.client_idx}_{cut_layer}"
            )
        else:
            client_lora_path = os.path.join(output_dir, f"global_lora_client_{cut_layer}")

        for path in [client_lora_path, server_lora_path, lm_head_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Could not find '{path}'. "
                    f"Run demo_train_SFLv2.py first with --cut_layer {cut_layer}, "
                    f"or pass --no_lora to evaluate the base model."
                )
        print(f"[{args.agg_method.upper()}] Client LoRA adapter : {client_lora_path}")

print(f"Server LoRA              : {server_lora_path}")
print(f"LM head                  : {lm_head_path}")

# =============================================================================
# Pretrained weights loader (same for all methods)
# =============================================================================
def load_multiple_safetensors(filenames):
    combined = {}
    for fn in filenames:
        combined.update(load_file(fn))
    return combined

file_paths = [f"{model_name}/model-0000{i}-of-00004.safetensors" for i in range(1, 5)]
pretrained_state_dict = load_multiple_safetensors(file_paths)
print(f"\nLoaded pretrained weights ({len(pretrained_state_dict)} tensors)")

# =============================================================================
# Weight loading helpers
# =============================================================================

def load_server_and_lmhead(server_model, lm_head, pretrained_sd, cut_layer):
    """
    Load server-side pretrained weights and lm_head.
    Identical for ALL aggregation methods — server is never touched by FLoRA.
    """
    state_dict    = {k.replace('model.', ''): v for k, v in pretrained_sd.items()}
    server_update = {}
    for key, value in state_dict.items():
        if 'layers.' in key:
            layer_num = int(key.split('.')[1])
            if layer_num >= cut_layer:
                new_key = key.replace(
                    f'layers.{layer_num}', f'layers.{layer_num - cut_layer}'
                )
                server_update[new_key] = value
        elif key == 'norm.weight':
            server_update[key] = value
    lm_head.weight.data = pretrained_sd['lm_head.weight'].to(torch.float32)
    server_model.load_state_dict(server_update, strict=False)
    print(f"  Server base weights loaded ({len(server_update)} tensors remapped)")


def load_client_pretrained(client_model, pretrained_sd, cut_layer):
    """
    Load client-side PRETRAINED weights.
    Used by FedAvg / FFA / FlexLoRA / no_lora.
    """
    state_dict    = {k.replace('model.', ''): v for k, v in pretrained_sd.items()}
    client_update = {}
    for key, value in state_dict.items():
        if 'layers.' in key:
            if int(key.split('.')[1]) < cut_layer:
                client_update[key] = value
        elif key == 'embed_tokens.weight':
            client_update[key] = value
    client_model.load_state_dict(client_update, strict=False)
    print(f"  Client pretrained weights loaded ({len(client_update)} tensors)")


def load_client_flora(client_model, flora_path, cut_layer):
    """
    Load client-side MERGED weights from a FLoRA safetensors file.
    Used ONLY by FLoRA — replaces load_client_pretrained for that method.

    flora_base_sd keys:  model.layers.{i}.self_attn.q_proj.weight  etc.
    client_model expects: layers.{i}.self_attn.q_proj.weight  (no 'model.' prefix)

    The function strips 'model.' and verifies the layer indices are all < cut_layer
    to catch any accidental key mismatches early.
    """
    flora_sd      = load_file(flora_path)
    client_update = {}
    skipped       = []

    for key, val in flora_sd.items():
        bare = key.replace('model.', '', 1)

        # Sanity: make sure we're not accidentally loading server-side layers
        m = re.search(r'layers\.(\d+)', bare)
        if m and int(m.group(1)) >= cut_layer:
            skipped.append(key)
            continue

        client_update[bare] = val

    if skipped:
        print(f"  [FLoRA WARNING] Skipped {len(skipped)} keys with layer_idx >= cut_layer "
              f"— these should not be in flora_base_sd. First: {skipped[0]}")

    client_model.load_state_dict(client_update, strict=False)
    print(f"  [FLoRA] Client merged weights loaded ({len(client_update)} tensors)")

    # Debug: show norms of first layer's q_proj and k_proj to confirm merge happened
    for key, val in sorted(client_update.items()):
        if 'layers.0' in key and ('q_proj' in key or 'k_proj' in key):
            norm = val.float().abs().sum().item()
            print(f"  [FLoRA DEBUG] {key}  |  norm={norm:.4f}")

# =============================================================================
# Model setup
# =============================================================================
from modelsplit_v2 import LlamaModel_Client, LlamaModel_Server

configuration = LlamaConfig.from_pretrained(model_name)
tokenizer     = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Llama-3: stop on eos OR <|eot_id|>
_eot_id = tokenizer.convert_tokens_to_ids('<|eot_id|>')
stop_ids = {tokenizer.eos_token_id}
if _eot_id and _eot_id != tokenizer.unk_token_id:
    stop_ids.add(_eot_id)

# Instantiate model skeletons
client_base = LlamaModel_Client(configuration, cut_layer)
server_base = LlamaModel_Server(configuration, cut_layer)
lm_head     = nn.Linear(configuration.hidden_size, configuration.vocab_size, bias=False)

# ── Server + lm_head: identical for ALL methods ───────────────────────────────
print("\nLoading server base weights...")
load_server_and_lmhead(server_base, lm_head, pretrained_state_dict, cut_layer)

# ── Client base weights: split by method ─────────────────────────────────────
print("\nLoading client base weights...")
if args.agg_method == 'flora' and not args.no_lora:
    # FLoRA: base weights already contain the merged LoRA deltas — no adapter needed
    load_client_flora(client_base, flora_base_path, cut_layer)
else:
    # FedAvg / FFA / FlexLoRA / no_lora: standard pretrained weights
    load_client_pretrained(client_base, pretrained_state_dict, cut_layer)

# Cast to BF16 — matches training precision
client_base.to(torch.bfloat16)
server_base.to(torch.bfloat16)
lm_head.to(torch.bfloat16)

# ── PEFT wrapping ─────────────────────────────────────────────────────────────
# lora_r must match max_lora_rank used during training.
# For FLoRA: we still wrap with PEFT so the model interface is identical,
# but we do NOT load any adapter — the LoRA weights are zeros (effectively absent).
# The merged base weights carry all the learned information.
peft_config = LoraConfig(
    r=args.lora_r,          # was hardcoded as 16; now matches training rank
    lora_alpha=args.lora_r,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj"],
)
if args.agg_method == "flora":
    client = client_base
else:
    client = get_peft_model(client_base,peft_config)
server_base.enable_input_require_grads = lambda: None
server = get_peft_model(server_base, peft_config)

# ── Load trained adapters ─────────────────────────────────────────────────────
print("\nLoading trained adapters...")

if not args.no_lora:
    if args.agg_method == 'flora':
        # FLoRA client: NO adapter to load — base weights already contain the delta.
        # Server adapter loads normally (server is never affected by FLoRA).
        print(f"  [FLoRA] Client: no LoRA adapter loaded "
              f"(delta already merged into base weights)")
        server.load_adapter(server_lora_path, "default")
        print(f"  Server LoRA adapter loaded from {server_lora_path}")
        lm_head.load_state_dict(torch.load(lm_head_path, map_location='cpu'))
        print(f"  LM head loaded from {lm_head_path}")
    else:
        # FedAvg / FFA / FlexLoRA: load client LoRA adapter normally
        client.load_adapter(client_lora_path, "default")
        print(f"  Client LoRA adapter loaded from {client_lora_path}")
        server.load_adapter(server_lora_path, "default")
        print(f"  Server LoRA adapter loaded from {server_lora_path}")
        lm_head.load_state_dict(torch.load(lm_head_path, map_location='cpu'))
        print(f"  LM head loaded from {lm_head_path}")
else:
    print("  --no_lora: running pretrained base model without any adapters")

# ── Move to GPU ───────────────────────────────────────────────────────────────
client.eval()
server.eval()
lm_head.eval()
client.to('cuda')
server.to('cuda')
lm_head.to('cuda')

# ── Mode string for logging ───────────────────────────────────────────────────
if args.no_lora:
    mode = "base (no LoRA)"
elif args.agg_method == 'flora':
    mode = f"flora (merged base weights, cut_layer={cut_layer})"
else:
    client_label = 'global_avg' if args.client_idx is None else str(args.client_idx)
    mode = f"{args.agg_method} | client={client_label} | lora_r={args.lora_r}"

print(f"\nModel ready — cut_layer={cut_layer} | {mode}")

# =============================================================================
# Dataset
# =============================================================================
gsm8k_test  = load_dataset("gsm8k", "main", split="test")
total_avail = len(gsm8k_test)
n_examples  = int(args.test_fraction * total_avail)
gsm8k_test  = gsm8k_test.select(range(n_examples))
print(f"Evaluating on {n_examples}/{total_avail} test examples\n")

def format_example(example):
    messages = [{"role": "user", "content": (
        "Solve the following math problem step by step.\n\n"
        f"Question: {example['question']}"
    )}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return {"text": prompt, "answer": example['answer']}

gsm8k_test = gsm8k_test.map(format_example)

# =============================================================================
# Answer extraction — unchanged
# =============================================================================
def extract_final_answer(text):
    m = re.search(r'####\s*([\d,]+)', text)
    if m:
        return m.group(1).replace(',', '')
    numbers = re.findall(r'\b\d[\d,]*\b', text)
    return numbers[-1].replace(',', '') if numbers else ""

# =============================================================================
# Greedy generation — unchanged
# The generate_answer function is identical for ALL methods because the model
# interface (client forward → server forward → lm_head) is the same regardless
# of how the weights were trained or loaded.
# =============================================================================
def generate_answer(prompt, max_new_tokens=256):
    input_ids  = tokenizer(
        prompt, return_tensors='pt', add_special_tokens=False
    ).input_ids.to('cuda')
    prompt_len = input_ids.shape[1]

    with torch.no_grad():
        for _ in range(max_new_tokens):
            attention_mask = torch.ones_like(input_ids)
            hidden_states, causal_mask, position_ids = client(
                input_ids=input_ids, attention_mask=attention_mask
            )
            server_out = server(
                hidden_states=hidden_states,
                causal_mask=causal_mask,
                position_ids=position_ids,
            )
            logits  = lm_head(server_out[0])
            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            if next_id.item() in stop_ids:
                break
            input_ids = torch.cat([input_ids, next_id], dim=-1)

    return tokenizer.decode(input_ids[0, prompt_len:], skip_special_tokens=True)

# =============================================================================
# Evaluation loop — unchanged
# =============================================================================
correct = 0
total   = len(gsm8k_test)

for example in tqdm(gsm8k_test, desc="Evaluating"):
    true_answer = extract_final_answer(example['answer'])
    generated   = generate_answer(example['text'], max_new_tokens=args.max_new_tokens)
    pred_answer = extract_final_answer(generated)
    if pred_answer == true_answer:
        correct += 1

print(f"\n{'='*50}")
print(f"Cut layer  : {cut_layer}")
print(f"Method     : {args.agg_method}")
print(f"Mode       : {mode}")
print(f"Accuracy   : {correct}/{total} = {correct/total:.4f}")
print(f"{'='*50}")
