# AkharFormer 🔤

**A Transformer built from scratch in PyTorch for English → Hindi translation.**

AkharFormer implements the full encoder-decoder architecture from *"Attention Is All You Need"* (Vaswani et al., 2017) end-to-end — embeddings, positional encoding, multi-head attention, residual connections, and the training loop — without relying on PyTorch's pre-built `nn.Transformer` module. Built as a learning project to understand the transformer architecture at the implementation level.

---

## ✨ Features

- Transformer encoder-decoder built layer-by-layer from scratch (no `nn.Transformer`)
- Custom multi-head self-attention and cross-attention
- Sinusoidal positional encoding
- Word-level tokenization trained from scratch on the dataset (via 🤗 `tokenizers`)
- Greedy decoding for inference/validation
- TensorBoard logging for training loss
- Checkpointing with resume support

---

## 🗂️ Project Structure

```
AkharFormer/
├── model.py           # Transformer architecture (attention, encoder, decoder, etc.)
├── dataset.py          # BilingualDataset + causal masking logic
├── config.py            # Hyperparameters and file path config
├── train.py            # Training loop, tokenizer building, validation
├── main.py               # Entry point / inference script
├── tokenizer_en.json     # Trained English tokenizer
├── tokenizer_hi.json     # Trained Hindi tokenizer
├── weights/              # Saved model checkpoints (.pt files)
├── runs/                 # TensorBoard logs
└── requirements.txt
```

---

## 🧠 Architecture

| Component | Detail |
|---|---|
| Model dimension (`d_model`) | 512 |
| Encoder/Decoder layers (`N`) | 6 |
| Attention heads (`h`) | 8 |
| Feed-forward dimension (`d_ff`) | 2048 |
| Dropout | 0.1 |
| Sequence length | Configurable (see `config.py`) |
| Optimizer | Adam (`eps=1e-9`) |
| Loss | Cross-Entropy with label smoothing (0.1) |

---

## 📚 Dataset

Trained on [`cfilt/iitb-english-hindi`](https://huggingface.co/datasets/cfilt/iitb-english-hindi) — the IIT Bombay English-Hindi Parallel Corpus (~1.6M sentence pairs), loaded via 🤗 `datasets`.

Tokenizers (`WordLevel`, trained from scratch, no pretrained subword vocab) are built once from the training data and cached as `tokenizer_en.json` / `tokenizer_hi.json`.

---

## ⚙️ Setup

```bash
git clone https://github.com/<your-username>/AkharFormer.git
cd AkharFormer
pip install -r requirements.txt
```

### Requirements
- Python 3.10+
- PyTorch
- 🤗 `datasets`
- 🤗 `tokenizers`
- `tensorboard`
- `tqdm`

---

## 🚀 Training

```bash
python train.py
```

Key settings live in `config.py`:

```python
{
    'batch_size': 8,
    'num_epochs': 20,
    'lr': 1e-4,
    'seq_len': 350,
    'd_model': 512,
    'lang_src': 'en',
    'lang_tgt': 'hi',
    ...
}
```

To resume training from a checkpoint, set `'preload'` to the epoch number (as a string) in `config.py`, e.g. `'preload': '05'`.

Training progress (loss) is logged to TensorBoard:

```bash
tensorboard --logdir runs
```

---

## ⚠️ Training Notes / Hardware Constraints

This model was trained on the full `cfilt/iitb-english-hindi` dataset (~1.6M sentence pairs). **Full local training was not feasible on a consumer laptop GPU** (NVIDIA RTX 2050, 4GB VRAM) due to VRAM and time constraints at the default `batch_size`/`seq_len` settings — training was instead run on **Google Colab's free-tier GPU**.

If you want to train locally on a smaller GPU:
- Reduce `batch_size` (e.g. `4`)
- Reduce `seq_len` (e.g. `160`)
- Train on a data subset instead of the full 1.6M pairs, e.g.:
  ```python
  ds_raw = ds_raw.shuffle(seed=42).select(range(30_000))
  ```

---

## 🔍 Inference

Inference isn't wrapped in a standalone script yet — translation currently happens via `greedy_decode()` inside `train.py`, which runs automatically during validation at the end of each training epoch. A dedicated CLI inference script is a planned addition..

---

## 📖 References

- Vaswani, A. et al. (2017). [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762)
- Kunchukuttan, A., Mehta, P., & Bhattacharyya, P. *The IIT Bombay English-Hindi Parallel Corpus*
- [🤗 `cfilt/iitb-english-hindi`](https://huggingface.co/datasets/cfilt/iitb-english-hindi)
- Umar Jamil — [*Coding a Transformer from scratch on PyTorch, with full explanation, training and inference*](https://www.youtube.com/watch?v=ISNdQcPhsts) — this project's implementation closely follows this walkthrough, adapted here for English → Hindi translation.

---

## 📝 License

This project is licensed under the [MIT License](LICENSE).

---

## 👤 Author

**Tushar Joshi**
Electrical Engineering, 2028 batch (AI minor) · Machine Learning Engineer enthusiast

- 💼 [LinkedIn](https://www.linkedin.com/in/tushar-joshi-47a5a9311)
- 💻 [GitHub](https://github.com/Tush2602)
- 📊 [Kaggle](https://www.kaggle.com/tush2602)
- 📄 [Resume](https://drive.google.com/file/d/1acaZ14eaTicIoxO3PinASp5dmXxcW3uX/view?usp=drive_link)

If you found this project useful or interesting, feel free to connect or drop a ⭐ on the repo!