# Aeitron 2-Hour Continuous Scratch Training on Modal.com

This directory contains the production-ready infrastructure to train the **Aeitron Defensive Cybersecurity AI Model** from scratch for **2 continuous hours** on high-performance GPUs (NVIDIA A100-80GB / H100) using **Modal.com**.

---

## 🚀 Quickstart: Running via Modal CLI (Option 1)

If you have the Modal CLI installed locally on your machine:

### 1. Authenticate with Modal
```bash
pip install modal
modal setup
```

### 2. Launch the 2-Hour Training Run on Cloud GPU
```bash
modal run deploy/modal/train_modal.py
```

To run with custom parameters (e.g. 1B model, different step count):
```bash
modal run deploy/modal/train_modal.py --profile 300m --steps 15000 --batch-size 4 --gradient-accumulation-steps 8
```

### 3. Automatic Checkpoint Persistence (`modal.Volume`)
All checkpoints, BPE tokenizer, binary token shards, and progress logs are saved automatically to the persistent volume `aeitron-training-volume`.

To list or download saved checkpoints:
```bash
modal volume ls aeitron-training-volume
modal volume get aeitron-training-volume train_output/checkpoints ./local_checkpoints
```

---

## 📓 Quickstart: Running in Modal Notebook / JupyterLab (Option 2)

If you prefer an interactive notebook interface:

1. Launch a Jupyter notebook session on Modal or open [Aeitron_Modal_Training.ipynb](file:///c:/Users/mah54/Desktop/Cyber_Security_AI_Architecture_Build/deploy/modal/Aeitron_Modal_Training.ipynb) in your cloud environment.
2. Run **Cell 1** to inspect GPU and CUDA VRAM.
3. Run **Cell 2 & 3** to ingest CISA KEV defensive advisories and build binary token shards.
4. Run **Cell 5** to execute the continuous 2-hour pretraining loop.
5. Run **Cell 6 & 7** to view the live loss convergence graph and test real-time defensive code patching!

---

## 🛡️ Error Recovery & Iteration Guide ("train korar somoi je problem hob eta thik kore samner dike agabo")

During long training runs, hardware or environment issues can occur. Here is how to diagnose and resolve them:

### 1. CUDA Out of Memory (OOM)
* **Symptom**: `RuntimeError: CUDA out of memory. Tried to allocate ...`
* **Fix**:
  1. Reduce `--batch-size` (e.g., from `4` to `2`).
  2. Increase `--gradient-accumulation-steps` (e.g., from `8` to `16`) to maintain the same effective global batch size.
  3. Ensure `--gradient-checkpointing` is enabled (it is enabled by default in `train_modal.py`).
  4. Ensure `--sequence-length` is within VRAM limits (e.g., `1024` or `2048`).

### 2. 2-Hour Session Timeout / Interruption
* **Symptom**: The Modal job reaches the 7200-second container limit or the network drops.
* **Fix**:
  * Checkpoints are automatically written every **250 steps**.
  * Simply re-run `modal run deploy/modal/train_modal.py`.
  * The script automatically detects previous checkpoints in `/vol/train_output/checkpoints/` and resumes training from the exact step where it stopped!

### 3. Loss Instability / Spikes
* **Symptom**: Loss suddenly spikes or returns NaN.
* **Fix**:
  * Aeitron employs RMSNorm, AdamW, and gradient clipping norm `1.0`.
  * If needed, lower the learning rate (`--learning-rate 1e-4`) or increase `--warmup-steps`.

### 4. Inspecting Logs in Real-Time
To tail progress while running:
```bash
modal app list
modal app logs <APP_ID>
```
Or inspect `/vol/logs/progress.jsonl` which records loss, learning rate, and throughput per step.
