#!/usr/bin/env python3
"""
visualize_finetuning_diff.py

ファインチューニングによる変化を可視化する総合スクリプト。

主な機能:
- 実行時に使用モデルを選択（prajjwal1/bert-tiny または huawei-noah/TinyBERT_General_4L_312D）
- ファインチューニング後モデルは finetuned_models/{model_name}/ に格納
- 可視化結果は analysis/visuals_{model_name}/ に出力
- Attention層およびFeed-Forward層の Before / After / 差分をヒートマップで可視化
- 各層のL2ノルム変化をプロット
- 埋め込み空間の変化をt-SNE/PCAで描画
"""

from __future__ import annotations
import argparse, os, re
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

try:
    from safetensors.torch import load_file as safe_load_file
    _HAVE_SAFETENSORS = True
except Exception:
    _HAVE_SAFETENSORS = False


# ===============================
# モデル選択プロンプト
# ===============================
def select_model_paths():
    print("\n🧠 使用するモデルを選択してください:\n")
    print("  [1] prajjwal1/bert-tiny  (2層, hidden=128, 超軽量)")
    print("  [2] huawei-noah/TinyBERT_General_4L_312D  (4層, hidden=312, 高精度)\n")

    choice = input("👉 モデル番号を入力してください [1/2]: ").strip()

    if choice == "2":
        base_name = "TinyBERT_General_4L_312D"
        pre = f"models/{base_name}/pytorch_model.bin"
        post = f"finetuned_models/{base_name}/pytorch_model.bin"
        emb_before = f"analysis/embeddings_before_{base_name}.npy"
        emb_after = f"analysis/embeddings_after_{base_name}.npy"
        outdir = f"analysis/visuals_{base_name}"
        print(f"\n✅ 選択: Huawei版 TinyBERT (4層) を使用します。\n")
    else:
        base_name = "bert-tiny"
        pre = f"models/{base_name}/pytorch_model.bin"
        post = f"finetuned_models/{base_name}/pytorch_model.bin"
        emb_before = f"analysis/embeddings_before_{base_name}.npy"
        emb_after = f"analysis/embeddings_after_{base_name}.npy"
        outdir = f"analysis/visuals_{base_name}"
        print(f"\n✅ 選択: praJJwal1/bert-tiny (2層) を使用します。\n")

    return base_name, pre, post, emb_before, emb_after, outdir


# ===============================
# モデル重みロード
# ===============================
def _load_state_dict(path: str) -> dict[str, torch.Tensor]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ モデルファイルが見つかりません: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext == ".bin":
        sd = torch.load(path, map_location="cpu")
        return sd["state_dict"] if "state_dict" in sd else sd
    elif ext in (".safetensors", ".safe"):
        if not _HAVE_SAFETENSORS:
            raise RuntimeError("safetensors not installed. pip install safetensors")
        return safe_load_file(path, device="cpu")
    else:
        raise ValueError(f"Unsupported file format: {ext}")


# ===============================
# 汎用描画関数（自動スケーリング対応）
# ===============================
def save_heatmap(name: str, arr: np.ndarray, outdir: str, cmap="bwr", vmin=None, vmax=None):
    plt.figure(figsize=(6, 4))

    if vmin is None or vmax is None:
        vmax = np.percentile(np.abs(arr), 99)
        vmin = -vmax

    plt.imshow(arr, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title(name)
    plt.tight_layout()

    os.makedirs(outdir, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
    out = os.path.join(outdir, f"{safe}.png")
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"[heatmap] {name} -> {out} (vmax={vmax:.5f})")


def plot_layer_deltas(deltas: list[float], outdir: str, title="Weight Change per Layer"):
    plt.figure(figsize=(6, 3))
    plt.plot(deltas, marker="o")
    plt.title(title)
    plt.xlabel("Layer Index")
    plt.ylabel("L2 Δ Weight")
    plt.tight_layout()
    out = os.path.join(outdir, "layer_deltas.png")
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"[plot] {title} -> {out}")


def visualize_embeddings(emb_before: str, emb_after: str, outdir: str):
    if not os.path.exists(emb_before) or not os.path.exists(emb_after):
        print("[skip] embedding files not found")
        return

    Xb, Xa = np.load(emb_before), np.load(emb_after)
    X = np.vstack([Xb, Xa])
    y = np.array(["Before"] * len(Xb) + ["After"] * len(Xa))

    if X.shape[1] > 50:
        X = PCA(n_components=50).fit_transform(X)
    X_2d = TSNE(n_components=2, random_state=42).fit_transform(X)

    plt.figure(figsize=(6, 6))
    plt.scatter(X_2d[y == "Before", 0], X_2d[y == "Before", 1], alpha=0.5, label="Before")
    plt.scatter(X_2d[y == "After", 0], X_2d[y == "After", 1], alpha=0.5, label="After")
    plt.legend()
    plt.title("Semantic Space (t-SNE): Before vs After")
    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir, "embedding_space_change.png"), dpi=200)
    plt.close()
    print("[plot] embedding_space_change.png")


# ===============================
# メイン処理
# ===============================
def main():
    base_name, pre, post, emb_before, emb_after, outdir = select_model_paths()

    # --- ファインチューニング後モデルが存在しない場合は警告 ---
    if not os.path.exists(post):
        print(f"\n⚠️ ファインチューニング済みモデルが存在しません: {post}")
        print(f"   → 先に train_and_export.py などでファインチューニングを実行してください。\n")
        return

    # --- モデル読み込み ---
    sd_pre = _load_state_dict(pre)
    sd_post = _load_state_dict(post)

    delta_per_layer = []

    for name, t_pre in sd_pre.items():
        if not isinstance(t_pre, torch.Tensor):
            continue
        if not re.search(r"encoder\.layer\.\d+\.", name):
            continue
        if name not in sd_post:
            continue

        t_post = sd_post[name]
        np_pre = t_pre.detach().cpu().numpy()
        np_post = t_post.detach().cpu().numpy()
        delta = np_post - np_pre

        # --- レイヤー別ディレクトリ作成 ---
        m = re.search(r"encoder\.layer\.(\d+)\.", name)
        layer_id = int(m.group(1)) if m else -1
        layer_dir = os.path.join(outdir, f"layer{layer_id}")
        os.makedirs(layer_dir, exist_ok=True)

        # --- 可視化 ---
        if re.search(r"(query|key|value)\.weight", name):
            save_heatmap(name + "_before", np_pre, layer_dir, cmap="viridis")
            save_heatmap(name + "_after", np_post, layer_dir, cmap="viridis")
            save_heatmap(name + "_delta", delta, layer_dir, cmap="bwr")
        elif re.search(r"(intermediate|output)\.dense\.weight", name):
            save_heatmap(name + "_before", np_pre, layer_dir, cmap="viridis")
            save_heatmap(name + "_after", np_post, layer_dir, cmap="viridis")
            save_heatmap(name + "_delta", delta, layer_dir, cmap="coolwarm")
            delta_per_layer.append(np.linalg.norm(delta))

    if delta_per_layer:
        plot_layer_deltas(delta_per_layer, outdir)

    if emb_before and emb_after:
        visualize_embeddings(emb_before, emb_after, outdir)


if __name__ == "__main__":
    main()
