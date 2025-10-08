#!/usr/bin/env python3
# ============================================
# TinyBERT / TinyBERT-4L ファインチューニング統合スクリプト
#
# このスクリプトは以下を実行します：
# 1. Triplet Loss による意味ベクトル学習 (SBERT構造)
# 2. ONNX形式でエクスポート（FP32）
# 3. INT8量子化（エッジ推論向け）
# 4. ファインチューニングなし（pretrained出力）モードにも対応
# ============================================

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType
import numpy as np
from tqdm import tqdm

# ==========================
# モデル選択
# ==========================
def select_model():
    print("\n🧠 対象モデルを選択してください:\n")
    print("  [1] prajjwal1/bert-tiny  (2層, hidden=128, 超軽量)")
    print("  [2] huawei-noah/TinyBERT_General_4L_312D  (4層, hidden=312, 高精度)\n")

    choice = input("👉 モデル番号を入力してください [1/2]: ").strip()

    if choice == "2":
        model_name = "huawei-noah/TinyBERT_General_4L_312D"
        base_dir = "models/TinyBERT_General_4L_312D"
        finetuned_dir = "finetuned_models/TinyBERT_General_4L_312D"
    else:
        model_name = "prajjwal1/bert-tiny"
        base_dir = "models/bert-tiny"
        finetuned_dir = "finetuned_models/bert-tiny"

    os.makedirs(finetuned_dir, exist_ok=True)
    return model_name, base_dir, finetuned_dir


# ==========================
# ファインチューニングを行うか選択
# ==========================
def ask_finetune():
    print("\n🧩 ファインチューニングを行いますか？")
    print("  [1] はい（Triplet Loss による学習を実行）")
    print("  [2] いいえ（pretrained モデルをそのままエクスポート）\n")
    return input("👉 選択してください [1/2]: ").strip() == "1"


# ==========================
# 共通設定
# ==========================
TRAIN_DATA_PATH = "data/train_triplets.txt"
MAX_LENGTH = 32
BATCH_SIZE = 16
EPOCHS = 5
LR = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==========================
# SBERT構造（mean pooling）
# ==========================
class SBERTEncoder(nn.Module):
    def __init__(self, bert_model):
        super().__init__()
        self.bert = bert_model

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = output.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        summed = (last_hidden * mask).sum(1)
        counts = mask.sum(1)
        mean_pooled = summed / counts
        return mean_pooled


# ==========================
# Tripletデータセット
# ==========================
class TripletDataset(Dataset):
    def __init__(self, path, tokenizer):
        self.samples = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                anchor, pos, neg = line.strip().split("\t")
                self.samples.append((anchor, pos, neg))
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        a, p, n = self.samples[idx]
        return self.tokenizer(
            [a, p, n],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH
        )


# ==========================
# Triplet Loss
# ==========================
def triplet_loss(anchor, positive, negative, margin=1.0):
    d_ap = (anchor - positive).pow(2).sum(1)
    d_an = (anchor - negative).pow(2).sum(1)
    return torch.relu(d_ap - d_an + margin).mean()


# ==========================
# ファインチューニング処理
# ==========================
def finetune_model(model_name, finetuned_dir):
    print("\n[1] モデルとトークナイザーを読み込み中...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModel.from_pretrained(model_name)
    model = SBERTEncoder(base_model).to(DEVICE)

    dataset = TripletDataset(TRAIN_DATA_PATH, tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    print("[2] ファインチューニング開始...")
    model.train()
    for epoch in range(EPOCHS):
        losses = []
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            input_ids = batch["input_ids"].squeeze(1).to(DEVICE)
            attention_mask = batch["attention_mask"].squeeze(1).to(DEVICE)
            a, p, n = input_ids[:, 0, :], input_ids[:, 1, :], input_ids[:, 2, :]
            am, pm, nm = attention_mask[:, 0, :], attention_mask[:, 1, :], attention_mask[:, 2, :]
            va, vp, vn = model(a, am), model(p, pm), model(n, nm)
            loss = triplet_loss(va, vp, vn)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())

        print(f"  ✅ Epoch {epoch+1}/{EPOCHS}  Loss: {np.mean(losses):.4f}")

    print("\n✅ ファインチューニング完了")

    model.bert.save_pretrained(finetuned_dir)
    tokenizer.save_pretrained(finetuned_dir)
    torch.save(model.state_dict(), os.path.join(finetuned_dir, "pytorch_model.bin"))
    return tokenizer, model


# ==========================
# ONNXエクスポート
# ==========================
def export_onnx(model, tokenizer, output_dir):
    print("\n[3] ONNXエクスポート中...")
    model.eval()

    onnx_fp32 = os.path.join(output_dir, "model_fp32.onnx")
    dummy_input_ids = torch.randint(0, tokenizer.vocab_size, (1, MAX_LENGTH), dtype=torch.long)
    dummy_attention_mask = torch.ones((1, MAX_LENGTH), dtype=torch.long)

    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask),
        onnx_fp32,
        input_names=["input_ids", "attention_mask"],
        output_names=["pooled_output"],
        dynamic_axes={"input_ids": {0: "batch"}, "attention_mask": {0: "batch"}},
        opset_version=13
    )
    print(f"✅ ONNXファイル出力完了 → {onnx_fp32}")
    return onnx_fp32


# ==========================
# キャリブレーションデータ
# ==========================
class DummyCalibReader(CalibrationDataReader):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.datas = [
            tokenizer(
                "災害発生時に避難を促す文章。",
                return_tensors="np",
                padding="max_length",
                truncation=True,
                max_length=MAX_LENGTH
            )
        ]
        self.index = 0

    def get_next(self):
        if self.index < len(self.datas):
            data = self.datas[self.index]
            self.index += 1
            return {
                "input_ids": data["input_ids"],
                "attention_mask": data["attention_mask"]
            }
        return None


# ==========================
# 量子化
# ==========================
def quantize_model(tokenizer, onnx_fp32, output_dir):
    print("\n[4] INT8量子化を実行中...")
    onnx_int8 = os.path.join(output_dir, "model_int8.onnx")
    quantize_static(
        model_input=onnx_fp32,
        model_output=onnx_int8,
        calibration_data_reader=DummyCalibReader(tokenizer),
        quant_format=QuantType.QUInt8
    )
    print(f"✅ INT8量子化完了 → {onnx_int8}")


# ==========================
# メイン
# ==========================
if __name__ == "__main__":
    model_name, base_dir, finetuned_dir = select_model()
    do_ft = ask_finetune()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModel.from_pretrained(model_name)
    model = SBERTEncoder(base_model).to(DEVICE)

    # ---- ファインチューニングを行う場合 ----
    if do_ft:
        tokenizer, model = finetune_model(model_name, finetuned_dir)
        output_dir = finetuned_dir
        print("\n✅ ファインチューニング済みモデルを出力します。")

    # ---- ファインチューニングを行わない場合 ----
    else:
        output_dir = base_dir
        os.makedirs(output_dir, exist_ok=True)
        print("\n✅ ファインチューニングなし: pretrainedモデルをそのまま出力します。")

    # ---- ONNX + 量子化共通処理 ----
    onnx_fp32 = export_onnx(model, tokenizer, output_dir)
    quantize_model(tokenizer, onnx_fp32, output_dir)

    print("\n🎯 すべての処理が完了しました。")
