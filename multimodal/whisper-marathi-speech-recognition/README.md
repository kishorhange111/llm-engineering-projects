# 04 · Fine-tuning Whisper for Marathi Speech Recognition (FLEURS)

Marathi has ~83 M speakers but is a *low-resource* language for speech AI. This project measures how far a small amount of in-language audio takes OpenAI's Whisper.

## Approach
| | |
|---|---|
| Model | **openai/whisper-small** (244 M params) — encoder–decoder Transformer on 80-channel log-Mel spectrograms |
| Data | Google **FLEURS** Marathi: **3,269 training utterances (11.9 h)**, 1,015 held-out test utterances (unseen speakers & sentences) |
| Training | full fine-tuning, language/task tokens fixed to `<|mr|><|transcribe|>`, 5 epochs, LR 1e-5, fp16, gradient checkpointing |
| Hardware | 1 × NVIDIA L4 · **22 min** |
| Metrics | **WER** (word error rate) and **CER** (character error rate — fairer for Marathi's rich morphology, where one wrong suffix makes a whole word "wrong") |

## Results (1,015 test utterances)
| Model | WER | CER |
|---|---|---|
| Whisper-small, zero-shot | 121.1 % | 53.6 % |
| **+ fine-tuning on 11.9 h of Marathi** | **45.3 %** | **14.7 %** |

**CER falls 3.6× (53.6 → 14.7 %)** from 22 minutes of training on one GPU. WER above 100 % for the original model means it *inserts* more words than the reference contains.

### What changed — real test utterances
| | |
|---|---|
| Reference | जे कोणी उंच भागात किंवा पर्वतांवरील रस्त्यावरून गाडी चालवणार आहेत त्यांनी हिम बर्फ किंवा गोठणबिंदूखालील तापमानाची शक्यता विचारात घ्यावी |
| Zero-shot | जे कोनी उंच भागात किवा परवतन वरिल रस्ते वरुन गाडी चलर नार आहेद त्यानी हिम, बर्फ किवा गोटन बिन्दू खालिल ताप्मानाची शक्किता विचाराद ग्यावी. |
| **Fine-tuned** | जे कोणी उंच भागात किंवा पर्वतांवरील रस्त्यावरुन गाडी चालवणार आहेत त्यांनी हिम बर्फ किंवा गोठण बिंधू खालील तापमानाची शक्यता विचारात घ्यावी |

| | |
|---|---|
| Reference | पोलिस अधीक्षक चंद्र शेखर सोलंकी यांनी सांगितले की आरोपी चेहरा झाकून घेऊन कोर्टात हजर झाला |
| Zero-shot | पूलिक्स अदिक्स्च्च्च्च्च्च्च्च्च्… *(repetition loop until the token limit)* |
| **Fine-tuned** | पोलिक्स अधिक्षकचंद्र शेखर सोळंकी यांनी साहितले की आरोपी चेहेरा जागून गेऊन कॉर्टा तदर झाला |

Three failure modes of the original model disappear: **repetition loops / hallucination**, **Hindi-style spellings** of Marathi words (किवा → किंवा, विचाराद → विचारात), and wrong vowel signs. Remaining errors are mostly near-homophones and word-boundary merges.

More examples in [`results/samples.json`](results/samples.json).

## Discussion points
- **Why CER matters here:** Marathi is agglutinative; a single wrong case suffix counts as a whole-word error in WER. CER shows the model now gets ~85 % of characters right.
- **Why the zero-shot model loops:** with little Marathi in pre-training, the decoder's language model is weak, so autoregressive decoding falls into high-probability repetition. Fine-tuning fixes the decoder's language model as much as the acoustics.
- **Engineering note:** the first run was input-bound (3 s/step, GPU mostly idle) because 80×3000 feature matrices were converted from Python lists in every batch; serving them as NumPy arrays straight from Arrow fixed it.
- **Next steps:** whisper-medium/large-v3 with LoRA, more data (Common Voice / IndicVoices), SpecAugment, and text normalisation before scoring (punctuation, digit vs. word numerals like 1972 vs. १९७६).

## Run
```bash
pip install torch transformers datasets jiwer soundfile librosa
python multimodal/whisper-marathi-speech-recognition/train.py      # ~35 min on an L4 incl. two full test-set evaluations
```
Data: FLEURS (Conneau et al., 2022), CC BY 4.0.
