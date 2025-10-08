# ================================================
# このスクリプトは以下の処理を行います：
# 1. ユーザーの入力文を意味ベクトル化（ONNX推論）
# 2. 事前にベクトル化された抽象メソッド群とコサイン類似度を計算
# 3. 最も近いメソッドを1つ選択して返す
# ================================================

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
import json
import os


# ==========================
# モデル選択関数
# ==========================
def select_model():
    print("\n🧠 使用するモデルを選択してください:\n")
    print("  [1] bert-tiny（未ファインチューニング）")
    print("  [2] finetuned bert-tiny（ファインチューニング済み）")
    print("  [3] TinyBERT_General_4L_312D（未ファインチューニング）")
    print("  [4] finetuned TinyBERT_General_4L_312D（ファインチューニング済み）\n")

    choice = input("👉 モデル番号を入力してください [1-4]: ").strip()

    # --- モデルごとのパス設定 ---
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
# 入力文 → 意味ベクトル（ONNX推論）
# ==========================
def get_embedding(text, tokenizer, session, max_length=32):
    # トークナイズ
    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="np"
    )

    # ✅ モデルが要求する入力名のみ抽出（token_type_idsエラー対策）
    valid_input_names = {i.name for i in session.get_inputs()}
    ort_inputs = {k: v for k, v in inputs.items() if k in valid_input_names}

    # ✅ モデルの期待するdtypeに合わせて自動キャスト
    expected_types = {i.name: i.type for i in session.get_inputs()}
    for k, v in ort_inputs.items():
        if "int64" in expected_types.get(k, ""):
            ort_inputs[k] = v.astype("int64")
        elif "int32" in expected_types.get(k, ""):
            ort_inputs[k] = v.astype("int32")
        elif "float" in expected_types.get(k, ""):
            ort_inputs[k] = v.astype("float32")

    # 推論実行
    embedding = session.run(["pooled_output"], ort_inputs)[0]
    return embedding[0]


# ==========================
# コサイン類似度の計算
# ==========================
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# ==========================
# メソッド選択（類似度順）
# ==========================
def select_best_method(user_text):
    # [1] モデル選択
    model_path, tokenizer_path, model_tag = select_model()

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ モデルファイルが見つかりません: {model_path}")

    # [2] モデル・トークナイザー読み込み
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    session = ort.InferenceSession(model_path)

    # [3] 入力文をベクトル化
    user_vec = get_embedding(user_text, tokenizer, session)

    # [4] ベクトルファイル読み込み
    METHODS_VECTORS_PATH = f"data/method_vectors_{model_tag}.npy"
    METHODS_TEXTS_PATH = f"data/method_texts_{model_tag}.json"

    if not os.path.exists(METHODS_VECTORS_PATH):
        raise FileNotFoundError(
            f"❌ ベクトルファイルが見つかりません: {METHODS_VECTORS_PATH}\n→ 先に gen_method_vectors.py をモデルごとに実行してください。"
        )

    method_vecs = np.load(METHODS_VECTORS_PATH)
    with open(METHODS_TEXTS_PATH, "r", encoding="utf-8") as f:
        method_texts = json.load(f)

    # [5] 類似度計算
    sims = [cosine_similarity(user_vec, v) for v in method_vecs]
    sorted_indices = np.argsort(sims)[::-1]

    # [6] 結果整形
    results = [{"method": method_texts[i], "score": float(sims[i])} for i in sorted_indices]
    return results


# ==========================
# 実行テスト
# ==========================
if __name__ == "__main__":
    user_input = "I just walked past my favorite ramen shop"
    results = select_best_method(user_input)

    print("\n[ユーザー入力]", user_input)
    print("\n[類似度スコア順メソッド一覧]")
    for i, r in enumerate(results, 1):
        print(f"{i}. スコア: {r['score']:.4f}")
        print(f"   メソッド: {r['method']}")
        print()
