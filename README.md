# TinyML Knowledge Distillation for Speech Emotion Recognition
*Lightweight Wav2Vec2 Student Models for Embedded AI*  
(Reference: KD for TinyML/Embedded AI Project)

## Overview
This repository presents an experimental study on applying Knowledge Distillation (KD) to compress the Wav2Vec2-base model into highly compact architectures suitable for TinyML deployment. Using the Toronto Emotional Speech Set (TESS), we evaluate how much performance can be retained when models are reduced from 95M parameters to 90K and 15K parameters.

## Motivation
Modern speech models achieve strong accuracy but require compute and memory far beyond what small, embedded devices can support. TinyML focuses on enabling inference under tight resource budgets, and KD provides a mechanism for transferring representational structure from a large teacher model to small student models. This project examines how distillation behaves when compressing models to sub–100K parameters.

## Dataset
**TESS (Toronto Emotional Speech Set)**  
Seven emotion classes, 2800 high-quality audio samples (.wav), balanced across categories. All audio is resampled to 16 kHz, normalized, and padded/truncated for consistent length.

## Models
### Teacher Model
- **Wav2Vec2-base** (95M parameters)  
- Fine-tuned for 7-class Speech Emotion Recognition  
- Achieved **98.39% accuracy**

### Student Models
#### Wav2Small
- 90K parameters  
- CNN-based VGG7-inspired architecture using LogMel spectrograms  
- Designed for edge deployment  
- Best performance: **94.82% accuracy**

#### Wav2Tiny
- 15K parameters  
- Further channel reductions and simplified pooling  
- Ultra-compact footprint for microcontroller-scale environments  
- Best performance: **86.25% accuracy**

## Knowledge Distillation
KD is implemented using:
- Softened teacher logits using temperature \( T \)
- Combined loss:  
  \( L = \alpha L_{CE} + (1 - \alpha)L_{KD} \)
- KL divergence for distillation loss

Ablation over \( T = \{1,2,3\} \) and \( \alpha = \{0,0.5,1\} \) shows:
- Wav2Small performs best at **α = 0.5, T = 2**
- Wav2Tiny benefits more from direct supervision and is less sensitive to softened teacher outputs

## Results
| Model          | Accuracy | Params |
|----------------|----------|--------|
| Wav2Vec2-base  | 98.39%   | 95M    |
| Wav2Small      | 94.82%   | 90K    |
| Wav2Tiny       | 86.25%   | 15K    |

Observations:
- Distillation retains competitive accuracy even with >1000× compression  
- Extremely small models (Wav2Tiny) show reduced ability to leverage teacher distributions

## Key Findings
- KD is effective for mid-size student models (~100K params)
- Very small models require careful tuning and may rely more on direct labels
- Temperature softening beyond \( T = 3 \) degrades learning stability
- Confusion matrices reveal predictable emotion-class ambiguities (e.g., Happy vs. Pleasant Surprise)

## Future Work
- Iterative KD: using Wav2Small as an intermediate teacher for Wav2Tiny  
- Exploring raw audio TinyML models in the style of TinyChirp  
- Microcontroller deployment and per-layer distillation strategies

## Reference PDF
`/mnt/data/KD for TinyML:Embedded AI_FR.pdf`

