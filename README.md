
<div align="right">
  <details>
    <summary >🌐 Language</summary>
    <div>
      <div align="center">
        <a href="https://openaitx.github.io/view.html?user=WtxwNs&project=BACH&lang=en">English</a>
        | <a href="https://openaitx.github.io/view.html?user=WtxwNs&project=BACH&lang=zh-CN">简体中文</a>
        | <a href="https://openaitx.github.io/view.html?user=WtxwNs&project=BACH&lang=zh-TW">繁體中文</a>
        | <a href="https://openaitx.github.io/view.html?user=WtxwNs&project=BACH&lang=ja">日本語</a>
        | <a href="https://openaitx.github.io/view.html?user=WtxwNs&project=BACH&lang=ko">한국어</a>
        | <a href="https://openaitx.github.io/view.html?user=WtxwNs&project=BACH&lang=hi">हिन्दी</a>
        | <a href="https://openaitx.github.io/view.html?user=WtxwNs&project=BACH&lang=th">ไทย</a>
        | <a href="https://openaitx.github.io/view.html?user=WtxwNs&project=BACH&lang=fr">Français</a>
        | <a href="https://openaitx.github.io/view.html?user=WtxwNs&project=BACH&lang=de">Deutsch</a>
        | <a href="https://openaitx.github.io/view.html?user=WtxwNs&project=BACH&lang=es">Español</a>
        | <a href="https://openaitx.github.io/view.html?user=WtxwNs&project=BACH&lang=it">Italiano</a>
        | <a href="https://openaitx.github.io/view.html?user=WtxwNs&project=BACH&lang=ru">Русский</a>
        | <a href="https://openaitx.github.io/view.html?user=WtxwNs&project=BACH&lang=pt">Português</a>
        | <a href="https://openaitx.github.io/view.html?user=WtxwNs&project=BACH&lang=nl">Nederlands</a>
        | <a href="https://openaitx.github.io/view.html?user=WtxwNs&project=BACH&lang=pl">Polski</a>
        | <a href="https://openaitx.github.io/view.html?user=WtxwNs&project=BACH&lang=ar">العربية</a>
        | <a href="https://openaitx.github.io/view.html?user=WtxwNs&project=BACH&lang=fa">فارسی</a>
        | <a href="https://openaitx.github.io/view.html?user=WtxwNs&project=BACH&lang=tr">Türkçe</a>
        | <a href="https://openaitx.github.io/view.html?user=WtxwNs&project=BACH&lang=vi">Tiếng Việt</a>
        | <a href="https://openaitx.github.io/view.html?user=WtxwNs&project=BACH&lang=id">Bahasa Indonesia</a>
        | <a href="https://openaitx.github.io/view.html?user=WtxwNs&project=BACH&lang=as">অসমীয়া</
      </div>
    </div>
  </details>

</div>

<p align="center">
  <img src="https://raw.githubusercontent.com/WtxwNs/BACH/main/tokenpair.png" width="85%"/>
  <br><br>
  <i>Watch how BACH turns raw tokens into structured music—step by step.</i>
</p>

# BACH: Bar-level AI Composing Helper  

<p align="center">
  <a href="https://arxiv.org/abs/2508.01394">
    <img src="https://img.shields.io/badge/arXiv-2508.01394-b31b1b.svg" alt="arXiv"/>
  </a>
  <a href="https://github.com/WtxwNs/BACH/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/WtxwNs/BACH" alt="License"/>
  </a>
  <img src="https://img.shields.io/github/repo-size/WtxwNs/BACH" alt="Repo Size"/>
  <img src="https://img.shields.io/github/stars/WtxwNs/BACH?style=social" alt="Stars"/>
</p>

&gt; *"Via Score to Performance: Efficient Human-Controllable Long Song Generation with Bar-Level Symbolic Notation"*  
&gt; ICASSP 2026 Submission – **Accepted**

---

## 🎼 One-sentence Summary  
BACH is the first **human-editable**, **bar-level** symbolic song generator:  
LLM writes lyrics → Transformer emits ABC score → off-the-shelf renderers give **minutes-long, Suno-level** music.  
**1 B params**, **minute-level** inference, **SOTA open-source**.

---

## 📦 What is inside this repo (preview release)
| Path | Description |
|------|-------------|
| `README.md` | This file |
| `code/` | inference code |
| `example.mp3` | an example song |
| `fig/` | Architecture figure |

---

## 🏗️ Model Architecture (one glance)

User prompt
Qwen3 — lyrics & style tags
BACH-1B Decoder-Only Transformer
ABC score (Dual-NTP + Chain-of-Score)
ABC → MIDI → FluidSynth + VOCALOID
Stereo mix


| Component | Key idea |
|-----------|----------|
| **Dual-NTP** | Predict `{vocal_patch, accomp_patch}` jointly every step |
| **Chain-of-Score** | Section tags `[START:Chorus] ... [END:Chorus]` for long coherence |
| **Bar-stream patch** | 16-char non-overlapping patches per bar |

---

## 🧪 Using this preview release

This checkout contains partial inference, preprocessing, and evaluation
utilities. It is **not a complete, CPU-ready BACH generation pipeline**.
The previously shown `bach/generate.py` entry point and root
`requirements.txt` are not present.

```bash
git clone https://github.com/WtxwNs/BACH.git
cd BACH

# Inspect the available inference options without loading a model.
python code/inference/infer.py --help

# Run dependency-free helper regression tests.
python -m unittest discover -s tests -v
```

The helper/CLI/shell tests use the standard library. Installing NumPy
also enables array-based stage-two and pitch-chunk regression tests;
those tests are skipped when NumPy is unavailable.

The inference dependency list is `code/requirements.txt`. Installing it
alone is insufficient for end-to-end generation: the referenced
`xcodec_mini_infer` modules and codec/decoder assets are not included.
The fine-tuning scripts also reference missing `scripts/train_lora.py`
and `core/tokenizer` components. Model execution, training, and audio
rendering require those assets and a compatible environment; no missing
weights, source, or datasets are supplied by this maintenance update.

Use only trusted model assets. Inference accepts the supported
`SoundStream` generator and loads tensor checkpoints with PyTorch's
restricted weights-only loader. Legacy pickled model objects are not
loaded automatically.

The pitch evaluator accepts explicit paths instead of machine-specific
directories; inspect its options with
`python code/evals/pitch_range/main.py --help`. Actual pitch extraction
still needs librosa, RMVPE, and its model checkpoint.

##  🎧 Listen now
example.mp3 is ready for you, it's a whole song. You can compare it with Suno🙂

## Full release upon related paper acceptance
- Complete training set (ABC + lyrics + structure labels)
- BACH-1B weights (Transformers format)
- Training scripts (multiphase + multitask + ICL)
- Complete Code

## 📎 Citation
Paper is released on Arxiv and IEEE Xplore, 
```bibtex
@INPROCEEDINGS{11463956,
  author={Wang, Tongxi and Yu, Yang and Wang, Qing and Qian, Junlang},
  booktitle={ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Via Score to Performance: Efficient Human-Controllable Long Song Generation with Bar-Level Symbolic Notation}, 
  year={2026},
  volume={},
  number={},
  pages={4856-4860},
  doi={10.1109/ICASSP55912.2026.11463956}}

}
