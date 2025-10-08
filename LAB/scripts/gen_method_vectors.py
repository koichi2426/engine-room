# ============================================
# このスクリプトは以下を実行します：
# 1. methods.txt に書かれた各抽象メソッドを読み込む
# 2. 選択したONNXモデルを使って各メソッド文をベクトル化
# 3. ベクトルを .npy 形式で保存（モデル別）
# 4. 元テキストを .json 形式で保存（表示・検索用）
# ============================================

import os
import json
import numpy as np
from transformers import AutoTokenizer
import onnxruntime as ort


# ==========================
# モデル選択関数（4種類対応）
# ==========================
def select_model():
    print("\n🧠 ベクトル化に使用するモデルを選択してください:\n")
    print("  [1] bert-tiny（未ファインチューニング）")
    print("  [2] finetuned bert-tiny（ファインチューニング済み）")
    print("  [3] TinyBERT_General_4L_312D（未ファインチューニング）")
    print("  [4] finetuned TinyBERT_General_4L_312D（ファインチューニング済み）\n")

    choice = input("👉 モデル番号を入力してください [1-4]: ").strip()

    # --- 選択ごとのパス設定 ---
    if choice == "1":
        model_dir = "models/bert-tiny"
        model_tag = "bert-tiny_pre"
        model_path = os.path.join(model_dir, "model_int8.onnx")
        print(f"\n✅ 選択: bert-tiny（未ファインチューニング）を使用します。\n")

    elif choice == "2":
        model_dir = "finetuned_models/bert-tiny"
        model_tag = "bert-tiny_ft"
        model_path = os.path.join(model_dir, "model_int8.onnx")
        print(f"\n✅ 選択: ファインチューニング済み bert-tiny を使用します。\n")

    elif choice == "3":
        model_dir = "models/TinyBERT_General_4L_312D"
        model_tag = "TinyBERT_4L_pre"
        model_path = os.path.join(model_dir, "model_int8.onnx")
        print(f"\n✅ 選択: TinyBERT（未ファインチューニング）を使用します。\n")

    else:
        model_dir = "finetuned_models/TinyBERT_General_4L_312D"
        model_tag = "TinyBERT_4L_ft"
        model_path = os.path.join(model_dir, "model_int8.onnx")
        print(f"\n✅ 選択: ファインチューニング済み TinyBERT (4層) を使用します。\n")

    tokenizer_path = model_dir
    return model_path, tokenizer_path, model_tag


# ==========================
# 設定
# ==========================
METHODS_TEXT_PATH = "data/methods.txt"
MAX_LENGTH = 32


# ==========================
# 単一文をONNXモデルでベクトル化
# ==========================
def encode(text, tokenizer, session):
    inputs = tokenizer(
        text,
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    )

    # ✅ モデルの入力仕様に合わせて自動調整
    valid_input_names = {i.name for i in session.get_inputs()}
    ort_inputs = {k: v for k, v in inputs.items() if k in valid_input_names}

    expected_types = {i.name: i.type for i in session.get_inputs()}
    for k, v in ort_inputs.items():
        if "int64" in expected_types.get(k, ""):
            ort_inputs[k] = v.astype("int64")
        elif "int32" in expected_types.get(k, ""):
            ort_inputs[k] = v.astype("int32")
        elif "float" in expected_types.get(k, ""):
            ort_inputs[k] = v.astype("float32")

    return session.run(["pooled_output"], ort_inputs)[0][0]


# ==========================
# メイン処理
# ==========================
def main():
    model_path, tokenizer_path, model_tag = select_model()

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ モデルファイルが見つかりません: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    session = ort.InferenceSession(model_path)

    os.makedirs("data", exist_ok=True)

    with open(METHODS_TEXT_PATH, encoding="utf-8") as f:
        method_lines = [line.strip() for line in f if line.strip()]

    print(f"[1] {len(method_lines)}件の抽象メソッドをベクトル化中 ({model_tag})...")

    vectors = np.array([encode(line, tokenizer, session) for line in method_lines])

    output_vec_path = f"data/method_vectors_{model_tag}.npy"
    output_text_path = f"data/method_texts_{model_tag}.json"

    np.save(output_vec_path, vectors)
    print(f"[2] ベクトルを保存しました → {output_vec_path}")

    with open(output_text_path, "w", encoding="utf-8") as jf:
        json.dump(method_lines, jf, ensure_ascii=False, indent=2)
    print(f"[3] メソッド原文を保存しました → {output_text_path}")


# ==========================
# 実行
# ==========================
if __name__ == "__main__":
    main()
