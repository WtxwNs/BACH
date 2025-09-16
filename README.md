# BACH: Bar-level AI Composing Helper  
&gt; *"Via Score to Performance: Efficient Human-Controllable Long Song Generation with Bar-Level Symbolic Notation"*  
&gt; ICASSP 2026 Submission – **Pending Review**

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

## 🧪 Quick start (CPU friendly)
```bash
# 1. Clone
git clone https://github.com/your-github/BACH.git
cd BACH

# 2. Install
pip install -r requirements.txt        # transformers>=4.41 mido abcpy fluidsynth

# 3. Generate ABC
python bach/generate.py \
    --prompt "A rainy-day lo-fi hip-hop song about missing the last train" \
    --out_abc demo/rainy_lofi.abc

# 4. Render audio
```

##  🎧 Listen now
example.mp3 is ready for you, it's a whole song — compare it with Suno

## Full release upon related paper acceptance
- Complete training set (ABC + lyrics + structure labels)
- BACH-1B weights (Transformers format)
- Training scripts (multiphase + multitask + ICL)
- Complete Code

## 📎 Citation
Paper is released on Arxiv, 
```bibtex
@misc{wang2025scoreperformanceefficienthumancontrollable,
      title={Via Score to Performance: Efficient Human-Controllable Long Song Generation with Bar-Level Symbolic Notation}, 
      author={Tongxi Wang and Yang Yu and Qing Wang and Junlang Qian},
      year={2025},
      eprint={2508.01394},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2508.01394}, 
}
