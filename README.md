# SFF : Split Federated Finetuning

SFF builds upon the foundation of [SPLITFM]([https://github.com/microsoft/LoRA](https://github.com/FDU-INC/SplitFM)) and Split Learning to enable privacy-preserving, parameter-efficient fine-tuning of foundation models. We introduce significant enhancements to support **heterogeneous client environments** and **Federated aggregation strategies**.

**Key Contributions:**

1. **Heterogeneous LoRA Ranks**: We have implemented the framework to support heterogeneous ranks across clients (e.g., `[4, 8, 16]`).
2. **Federated Aggregation Schemes**: We have integrated four distinct aggregation methods to handle the updates from these heterogeneous clients:
    * **Average**: Performs naive element-wise averaging of adapter matrices, though this can introduce cross-term interference that may destabilize the split forward pass.
    * **Freeze**: Freezes one projection matrix while training only the other to enforce linearity and eliminate cross-term noise, at the cost of reduced trainable parameters.
    * **Stack**: Concatenates adapters to preserve distinct client subspaces without information loss, though this increases the global rank linearly with the number of clients, leading to higher communication costs.
    * **SVD**: Aggregates updates in the full update space and re-projects via SVD to maintain a fixed rank, balancing noise reduction with computational efficiency by identifying principal update directions.

This framework currently supports PyTorch-based GPT-2 models and **LLaMA-based models**, with plans to integrate more open-source LLMs in the future.

---

### User Guide:

#### 1. Build

##### 1.1 Environment Requirements

We have verified in the environment below:

+ OS: Ubuntu 22.04
+ Python: 3.10.0

##### 1.2 Installation

1. Clone the repo and set up the environment.

```bash
conda create -n SFF python=3.10 -y
conda activate SFF
```

2. Navigate to the examples directory and install the required packages.

```bash
cd SFF/examples
pip install -r requirements.txt
```

3. Download the necessary pre-trained models, datasets, and evaluation scripts.

```bash
# Download pre-trained GPT-2 checkpoints
bash download_pretrained_checkpoints.sh

# Prepare datasets
bash create_datasets.sh

# Download evaluation scripts
cd ./eval
bash download_evalscript.sh
cd ..
```

---

#### 2. SFF Module Libraries

##### 2.1 Repository

Our implementation is based on the fine-tuning code for GPT-2 in Hugging Face.
There are several directories in this repo:

* `src/` contains the source code used for data processing, training, and decoding.
* `eval/` contains the code for task-specific evaluation scripts.
* `data/` contains the raw data we used in our experiments.
* `vocab/` contains the GPT-2 vocabulary files.

##### 2.2 Key Hyper-Parameters

| Argument | Description | Default/Example |
|---|---|---|
| `--train_batch_size` | Training batch size. | 4 |
| `--grad_acc` | Number of gradient accumulation steps. | 1 |
| `--seq_len` | Sequence length. | 512 |
| `--model_card` | Path to the model configuration file. | ${MODEL_CARD} |
| `--init_checkpoint` | Path to the initial checkpoint file. | gpt2-large-pytorch_model.bin |
| `--platform` | Execution platform. | local |
| `--lr` | Learning rate. | 0.0002 |
| `--max_epoch` | Maximum number of training epochs. | 1 |
| `--lora_dim` | The dimension of LoRA. | ${LORA_DIM} |
| `--lora_alpha` | Alpha hyperparameter for LoRA. | 32 |
| `--cut_layer` | The layer index where the model is split. | ${CUT_LAYER} |
| `--agg_method` | Aggregation method (e.g., stack, svd, avg, freeze). | ${AGG_METHOD} |
| `--lora_ranks` | List of heterogeneous ranks for clients. | ${LORA_RANKS} |
| `--work_dir` | Working directory for saving models/logs. | . |

---

#### 3. LLaMA Extension

In addition to the GPT-2 pipeline, we have extended SFF to support **LLaMA-based models**. Three new files have been added to the `SFF/` directory to enable split federated fine-tuning and evaluation on LLaMA:

| File | Purpose |
|---|---|
| `llama_modelsplit.py` | Server-side model splitting logic for LLaMA |
| `llama_SFF.py` | Federated aggregation schemes adapted for LLaMA |
| `llama_eval.py` | GSM8K evaluation harness for trained LLaMA models |

---

##### 3.1 `llama_modelsplit.py` — Server-Side Model Splitting

This file implements the server-side **model splitting logic** for LLaMA architectures. It defines how the LLaMA model is partitioned across the cut layer, separating the server-side (upper) and client-side (lower) portions of the network to enable split learning in the federated setting.

**Key functionality:**
- Splits a LLaMA model at a configurable `--cut_layer`
- Manages the forward pass handoff between the client and server sub-models
- Compatible with the cut layer sweep driven by SLURM array jobs (e.g. `CUT_LAYER = 1 + SLURM_ARRAY_TASK_ID * 3`)

---

##### 3.2 `llama_SFF.py` — Federated Aggregation for LLaMA

This file contains the **federated fine-tuning loop and aggregation schemes** adapted for LLaMA models. It mirrors the core SFF training pipeline but is tailored to LLaMA's architecture and tokenizer, and supports heterogeneous LoRA ranks across clients.

**Key functionality:**
- Implements all four aggregation methods (`fedavg`, `ffa`, `flora`, `flexlora`) for LLaMA LoRA adapters
- Supports a configurable number of clients, client sampling fraction, and local training epochs per round
- Handles both uniform (`--lora_r`) and heterogeneous (`--lora_ranks`) rank configurations
- Coordinates split forward and backward passes between the client and server sub-models defined in `llama_modelsplit.py`

---

##### 3.3 `llama_eval.py` — GSM8K Evaluation for LLaMA

This file provides an **evaluation harness** for assessing the performance of fine-tuned LLaMA models on the [GSM8K](https://github.com/openai/grade-school-math) benchmark, a dataset of grade-school math word problems used to measure reasoning capability.

**Key functionality:**
- Loads a trained LLaMA checkpoint from `--output_dir`
- Runs inference on the GSM8K test set at a given `--cut_layer`
- Computes accuracy by parsing and comparing model-generated answers against ground-truth solutions

---

##### 3.4 Key Hyper-Parameters

| Argument | Description | Default/Example |
|---|---|---|
| `--num_clients` | Total number of federated clients. | 30 |
| `--client_fraction` | Fraction of clients sampled per round. | 0.1 |
| `--rounds` | Number of federated communication rounds. | 50 |
| `--local_epochs` | Number of local training epochs per round. | 1 |
| `--cut_layer` | The layer index where the model is split. | ${CUT_LAYER} |
| `--agg_method` | Aggregation method (`fedavg`, `ffa`, `flora`, `flexlora`). | ${AGG_METHOD} |
| `--lora_r` | Uniform LoRA rank (used if `--lora_ranks` is not set). | 16 |
| `--lora_ranks` | Comma-separated heterogeneous ranks per client (overrides `--lora_r`). | ${LORA_RANKS} |
| `--seed` | Random seed for reproducibility. | 42 |
| `--output_dir` | Directory for saving model checkpoints and logs. | ${OUTPUT_DIR} |

---

#### 4. LLaMA Training Process

##### 1. Train LLaMA with Heterogeneous SFF

Run the following command to start training. Ensure you set your environment variables (like `AGG_METHOD`, `CUT_LAYER`, `OUTPUT_DIR`, etc.) before running.

```bash
python examples/NLG_llama3/demo_train_SFLv3.py \
    --num_clients ${NUM_CLIENTS} \
    --client_fraction ${CLIENT_FRACTION} \
    --rounds ${ROUNDS} \
    --local_epochs ${LOCAL_EPOCHS} \
    --cut_layer ${CUT_LAYER} \
    --output_dir ${OUTPUT_DIR} \
    --agg_method ${AGG_METHOD} \
    --seed ${SEED} \
    --lora_r ${LORA_R}
    # Replace --lora_r with --lora_ranks ${LORA_RANKS} for heterogeneous ranks
```

##### 2. Evaluate on GSM8K

Evaluate the trained LLaMA model on the GSM8K benchmark.

```bash
python examples/NLG_llama3/evaluation_gsm8k_v2.py \
    --cut_layer ${CUT_LAYER} \
    --output_dir ${OUTPUT_DIR}
```

---

#### 5. GPT-2 Training Process

##### 1. Train GPT-2 with Heterogeneous SFF

Run the following command to start training. Ensure you set your environment variables (like `MODEL_CARD`, `LORA_DIM`, `AGG_METHOD`, etc.) before running.

```bash
python -m torch.distributed.launch --nproc_per_node=1 --master_port=$((RANDOM%10000+20000)) --use_env src/gpt2_ft_sfl.py \
    --train_data0 ./data/e2e/train0.jsonl \
    --train_data1 ./data/e2e/train1.jsonl \
    --train_data2 ./data/e2e/train2.jsonl \
    --valid_data ./data/e2e/valid.jsonl \
    --train_batch_size 4 \
    --grad_acc 1 \
    --valid_batch_size 4 \
    --seq_len 512 \
    --model_card ${MODEL_CARD} \
    --init_checkpoint ./pretrained_checkpoints/gpt2-large-pytorch_model.bin \
    --platform local \
    --clip 0.0 \
    --lr 0.0002 \
    --weight_decay 0.01 \
    --correct_bias \
    --adam_beta2 0.999 \
    --scheduler linear \
    --warmup_step 500 \
    --max_epoch 1 \
    --save_interval 999999 \
    --lora_dim ${LORA_DIM} \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --label_smooth 0.1 \
    --work_dir "." \
    --random_seed ${SEED} \
    --cut_layer ${CUT_LAYER} \
    --agg_method ${AGG_METHOD} \
    --lora_ranks ${LORA_RANKS}
```

##### 2. Generate Outputs (Inference)

Use beam search to generate outputs from the trained model.

```bash
python -m torch.distributed.launch --nproc_per_node=1 --master_port=$((RANDOM%10000+20000)) --use_env src/gpt2_beam.py \
    --data ./data/e2e/test.jsonl \
    --batch_size 8 \
    --seq_len 512 \
    --eval_len 64 \
    --model_card ${MODEL_CARD} \
    --init_checkpoint "$CHECKPOINT_PATH" \
    --platform local \
    --lora_dim ${CURRENT_RANK} \
    --lora_alpha 32 \
    --beam 5 \
    --length_penalty 0.8 \
    --no_repeat_ngram_size 4 \
    --repetition_penalty 1.0 \
    --eos_token_id 628 \
    --work_dir "${JOB_WORK_DIR}" \
    --output_file "${PREDICT_FILE}"
```

##### 3. Decode Outputs

Convert the generated JSONL outputs into flat text files for evaluation.

```bash
python src/gpt2_decode.py \
    --vocab ./vocab \
    --sample_file "${JOB_WORK_DIR}/${PREDICT_FILE}" \
    --input_file ./data/e2e/test_formatted.jsonl \
    --output_ref_file "${REF_FILE}" \
    --output_pred_file "${PRED_FILE}"
```

##### 4. Run Evaluation

Evaluate the decoded predictions against the reference file.

```bash
python eval/e2e/measure_scores.py "${REF_FILE}" "${PRED_FILE}" -p
```
