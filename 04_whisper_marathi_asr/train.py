"""
Fine-tuning Whisper for Marathi speech recognition (FLEURS)
===========================================================

TASK: Marathi speech -> Marathi text. Marathi has ~83M speakers but is a
low-resource language for speech AI: general-purpose models transcribe it
poorly. This project measures how much a small amount of in-language data
helps.

MODEL: OpenAI Whisper-small (244M params) - an encoder-decoder Transformer.
    * Encoder: 30 s of audio -> 80-channel log-Mel spectrogram -> Transformer.
    * Decoder: generates text tokens, conditioned on special tokens that set
      the language (<|mr|>) and the task (<|transcribe|>).
    Whisper was pre-trained on 680k hours of multilingual audio, but very
    little of it Marathi.

FINE-TUNING: full fine-tuning (seq2seq, teacher forcing) on the FLEURS
Marathi training split, fp16, with the language/task tokens fixed.

METRICS on the FLEURS Marathi test split (held-out speakers/sentences):
    WER - word error rate  = (substitutions + deletions + insertions) / words
    CER - character error rate (fairer for Marathi's rich morphology:
          one wrong suffix makes a whole word "wrong" under WER)
    Measured for the original model (zero-shot) and after fine-tuning.
"""
import io
import json
import os
import time

import numpy as np
import torch

QUICK = os.environ.get("QUICK") == "1"
MODEL = os.environ.get("MODEL", "openai/whisper-small")
OUT = os.environ.get("OUT", "/content/whisper_mr")
EPOCHS = 1 if QUICK else float(os.environ.get("EPOCHS", "5"))
N_TEST = 40 if QUICK else None
BATCH = 16
LR = 1e-5
TIME_BUDGET_HOURS = float(os.environ.get("TIME_BUDGET_HOURS", "1.0"))


def load_split(split, limit=None):
    """FLEURS Marathi, audio decoded ourselves (16 kHz mono float32) to avoid extra audio backends."""
    import soundfile as sf
    from datasets import Audio, load_dataset

    ds = load_dataset("google/fleurs", "mr_in", split=split)
    ds = ds.cast_column("audio", Audio(decode=False))
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    def decode(ex):
        a = ex["audio"]
        wav, sr = sf.read(io.BytesIO(a["bytes"])) if a.get("bytes") else sf.read(a["path"])
        if wav.ndim > 1:
            wav = wav.mean(1)
        if sr != 16000:
            import librosa
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        return {"wav": wav.astype("float32"), "text": ex["transcription"]}

    return ds.map(decode, remove_columns=ds.column_names)


def wer_cer(refs, hyps):
    import jiwer

    return float(jiwer.wer(refs, hyps)), float(jiwer.cer(refs, hyps))


@torch.no_grad()
def transcribe(model, processor, ds, label):
    model.eval()
    hyps, t = [], time.time()
    for i in range(0, len(ds), 32):
        batch = ds[i:i + 32]
        feats = processor.feature_extractor(batch["wav"], sampling_rate=16000, return_tensors="pt").input_features
        ids = model.generate(feats.to(model.device, model.dtype), language="marathi", task="transcribe",
                             max_new_tokens=225)
        hyps += processor.batch_decode(ids, skip_special_tokens=True)
    refs = list(ds["text"])
    wer, cer = wer_cer(refs, [h.strip() for h in hyps])
    res = {"wer": wer, "cer": cer, "minutes": (time.time() - t) / 60}
    print(label, res, flush=True)
    return res, list(zip(refs[:10], hyps[:10]))


def main():
    from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainerCallback,
                              WhisperForConditionalGeneration, WhisperProcessor)

    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(42)
    train = load_split("train", 200 if QUICK else None)
    test = load_split("test", N_TEST)
    hours = sum(len(w) for w in train["wav"]) / 16000 / 3600
    print(f"train {len(train)} utterances ({hours:.1f} h), test {len(test)}")

    processor = WhisperProcessor.from_pretrained(MODEL, language="marathi", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(MODEL).to("cuda")
    model.generation_config.language = "marathi"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    base_res, base_samples = transcribe(model.half(), processor, test, "zero-shot")
    model.float()

    def prepare(ex):
        ex["input_features"] = processor.feature_extractor(ex["wav"], sampling_rate=16000).input_features[0]
        ex["labels"] = processor.tokenizer(ex["text"]).input_ids   # includes <|startoftranscript|><|mr|><|transcribe|>...
        return ex

    train_feats = train.map(prepare, remove_columns=["wav", "text"])
    # Serve the 80x3000 log-Mel features as NumPy arrays straight from Arrow. As Python lists they had to be
    # converted element by element in every batch, which left the GPU idle (~3 s/step instead of <1 s).
    train_feats = train_feats.with_format("numpy", columns=["input_features"], output_all_columns=True)

    def collate(features):
        x = processor.feature_extractor.pad([{"input_features": f["input_features"]} for f in features],
                                            return_tensors="pt")
        y = processor.tokenizer.pad([{"input_ids": f["labels"]} for f in features], return_tensors="pt")
        labels = y["input_ids"].masked_fill(y.attention_mask.ne(1), -100)
        if (labels[:, 0] == model.config.decoder_start_token_id).all():
            labels = labels[:, 1:]          # the model prepends the start token itself
        x["labels"] = labels
        return x

    class TimeBudget(TrainerCallback):
        def __init__(self):
            self.deadline = time.time() + TIME_BUDGET_HOURS * 3600

        def on_step_end(self, args, state, control, **kw):
            if time.time() > self.deadline:
                control.should_training_stop = True
            return control

    args = Seq2SeqTrainingArguments(
        output_dir=OUT, num_train_epochs=EPOCHS, per_device_train_batch_size=BATCH, learning_rate=LR,
        warmup_steps=0.1, gradient_checkpointing=True, fp16=True, logging_steps=25, save_strategy="no",
        report_to="none", remove_unused_columns=False, seed=42)
    trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=train_feats, data_collator=collate,
                             callbacks=[TimeBudget()])
    t = time.time()
    result = trainer.train()
    train_min = (time.time() - t) / 60

    tuned_res, tuned_samples = transcribe(model.half(), processor, test, "fine-tuned")
    summary = {"model": MODEL, "train_utterances": len(train), "train_hours": hours, "test_utterances": len(test),
               "epochs": EPOCHS, "train_minutes": train_min, "final_train_loss": result.training_loss,
               "gpu": torch.cuda.get_device_name(0), "zero_shot": base_res, "fine_tuned": tuned_res}
    json.dump(summary, open(os.path.join(OUT, "metrics.json"), "w"), indent=2)
    json.dump({"zero_shot": base_samples, "fine_tuned": tuned_samples},
              open(os.path.join(OUT, "samples.json"), "w"), ensure_ascii=False, indent=2)
    json.dump(trainer.state.log_history, open(os.path.join(OUT, "log_history.json"), "w"))
    print(json.dumps(summary, indent=2))
    print("DONE")


if __name__ == "__main__":
    main()
