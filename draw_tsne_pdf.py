# draw_tsne.py / plot_tsne_trajs.py
# -*- coding: utf-8 -*-

import os
import sys
import json
import math
import glob
import argparse
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import inspect
import sklearn

import numpy as np

# YAML：ruamel 优先，退化到 PyYAML
try:
    from ruamel.yaml import YAML
    _YAML_IMPL = "ruamel"
except Exception:
    import yaml as _pyyaml
    _YAML_IMPL = "pyyaml"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# =========================
# 配置数据结构
# =========================
@dataclass
class RunCfg:
    name: str
    traj_dir: str
    color: Optional[str] = None
    include_episodes: Optional[List[int]] = None
    max_eps: Optional[int] = None


@dataclass
class PlotCfg:
    key: str                       # obs_flat | a_env | a_arm | obs_act | z_obs | z_act | z_obs_act
    mode: str = "per-step"         # per-step | per-episode-mean
    out_dir: str = "./tsne_out"
    seed: int = 42

    # 采样 / 预处理
    max_per_traj: int = 200        # per-step 时每条轨最多抽多少步
    use_pca: bool = True
    pca_dim: int = 50
    coral: bool = False            # 是否做 CORAL（二阶统计对齐）
    balance_by_label: bool = True
    max_samples_per_label: Optional[int] = None  # 平衡上限，None 表示不截断

    # t-SNE 参数
    perplexity: Optional[float] = 30.0
    n_iter: int = 1500
    metric: str = "euclidean"      # euclidean | cosine
    learning_rate: str = "auto"
    init: str = "pca"
    dpi: int = 160

    # 可视化
    point_size: float = 6.0
    alpha: float = 0.4
    legend_loc: str = "best"


# =========================
# YAML 读取
# =========================
def load_yaml(path: str) -> dict:
    if _YAML_IMPL == "ruamel":
        yaml = YAML(typ="safe")
        with open(path, "r", encoding="utf-8") as f:
            return yaml.load(f)
    else:
        with open(path, "r", encoding="utf-8") as f:
            return _pyyaml.safe_load(f)


def parse_config(cfg_path: str) -> Tuple[PlotCfg, List[RunCfg]]:
    raw = load_yaml(cfg_path)
    if "plot" not in raw or "runs" not in raw:
        raise ValueError("YAML 缺少必要字段：plot / runs")

    p = raw["plot"]
    plot = PlotCfg(
        key=str(p.get("key")),
        mode=str(p.get("mode", "per-step")),
        out_dir=str(p.get("out_dir", "./tsne_out")),
        seed=int(p.get("seed", 42)),
        max_per_traj=int(p.get("max_per_traj", 200)),
        use_pca=bool(p.get("use_pca", True)),
        pca_dim=int(p.get("pca_dim", 50)),
        coral=bool(p.get("coral", False)),
        balance_by_label=bool(p.get("balance_by_label", True)),
        max_samples_per_label=p.get("max_samples_per_label", None),
        perplexity=p.get("perplexity", 30.0),
        n_iter=int(p.get("n_iter", 1500)),
        metric=str(p.get("metric", "euclidean")),
        learning_rate=str(p.get("learning_rate", "auto")),
        init=str(p.get("init", "pca")),
        dpi=int(p.get("dpi", 160)),
        point_size=float(p.get("point_size", 6.0)),
        alpha=float(p.get("alpha", 0.4)),
        legend_loc=str(p.get("legend_loc", "best")),
    )
    if plot.max_samples_per_label is not None:
        plot.max_samples_per_label = int(plot.max_samples_per_label)

    runs = []
    for r in raw["runs"]:
        runs.append(
            RunCfg(
                name=str(r["name"]),
                traj_dir=str(r["traj_dir"]),
                color=r.get("color", None),
                include_episodes=r.get("include_episodes", None),
                max_eps=r.get("max_eps", None),
            )
        )
    return plot, runs


# =========================
# 数据加载与采样
# =========================
def _ensure_traj_dir(path: str) -> str:
    """允许传 eval_xxx 或 eval_xxx/traj_npz，统一返回 traj_npz 路径。"""
    if os.path.isdir(os.path.join(path, "traj_npz")):
        return os.path.join(path, "traj_npz")
    return path  # 用户已直接指到 traj_npz


def _list_npz(traj_dir: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(traj_dir, "*.npz")))
    return files


def _episode_id_from_npz(npz_path: str) -> Optional[int]:
    base = os.path.basename(npz_path)
    # 兼容 ep_003.npz / ep_003_xxx.npz
    for token in base.replace(".", "_").split("_"):
        if token.isdigit():
            try:
                return int(token)
            except Exception:
                pass
    return None


def _uniform_indices(T: int, max_k: int) -> np.ndarray:
    """从 [0..T-1] 均匀采样最多 max_k 个索引（包含两端）"""
    if T <= max_k:
        return np.arange(T, dtype=np.int64)
    xs = np.linspace(0, T - 1, num=max_k)
    idx = np.unique(np.round(xs).astype(np.int64))
    return idx


def _load_points_from_npz(npz_path: str, key: str, mode: str,
                          max_per_traj: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    返回:
      X: (N, D)
      meta: (N, 3) -> [episode_id, step_idx, T_original]
            per-episode-mean 模式下 step_idx = -1
    """
    data = np.load(npz_path, allow_pickle=True)
    if key not in data.files:
        raise KeyError(f"{os.path.basename(npz_path)} 不含 key='{key}'。可用: {list(data.files)}")

    arr = data[key]
    if arr.ndim == 1:
        arr = arr[None, :]

    T, D = arr.shape[0], arr.shape[-1]

    ep = data["__episode__"].item() if "__episode__" in data.files else _episode_id_from_npz(npz_path)
    if ep is None:
        ep = -1

    if mode == "per-step":
        idx = _uniform_indices(T, max_per_traj)
        X = arr[idx]
        meta = np.stack([np.full_like(idx, ep), idx, np.full_like(idx, T)], axis=1)
        return X.astype(np.float32), meta.astype(np.int64)

    elif mode == "per-episode-mean":
        X = np.mean(arr, axis=0, keepdims=True)
        meta = np.array([[ep, -1, T]], dtype=np.int64)
        return X.astype(np.float32), meta

    else:
        raise ValueError(f"未知 mode: {mode}")


def _concat_with_label(
    all_blocks: List[Tuple[np.ndarray, np.ndarray, str]],
    key_name: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    all_blocks: list of (X, meta, label_str)
    返回:
      X_all:   (N, D)
      meta_all:(N, 3)
      y_all:   (N,) -> 字符串标签
    """
    if not all_blocks:
        return (
            np.empty((0, 0), dtype=np.float32),
            np.empty((0, 3), dtype=np.int64),
            np.empty((0,), dtype=object),
        )

    # 维度一致性
    dims = [blk[0].shape[1] for blk in all_blocks]
    if len(set(dims)) != 1:
        detail = [(i, blk[0].shape) for i, blk in enumerate(all_blocks)]
        raise RuntimeError(f"不同 npz 的 '{key_name}' 维度不一致: {detail}")

    X_all = np.concatenate([blk[0] for blk in all_blocks], axis=0)
    meta_all = np.concatenate([blk[1] for blk in all_blocks], axis=0)
    y_all = np.concatenate([np.array([blk[2]] * blk[0].shape[0], dtype=object) for blk in all_blocks], axis=0)
    return X_all, meta_all, y_all


def _balance_by_label(
    X: np.ndarray,
    meta: np.ndarray,
    y: np.ndarray,
    max_per_label: Optional[int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if max_per_label is None:
        return X, meta, y
    out_X, out_meta, out_y = [], [], []
    labels = np.unique(y)
    for lb in labels:
        idx = np.where(y == lb)[0]
        if idx.size > max_per_label:
            sel = np.random.choice(idx, size=max_per_label, replace=False)
        else:
            sel = idx
        out_X.append(X[sel])
        out_meta.append(meta[sel])
        out_y.append(y[sel])
    return np.concatenate(out_X, 0), np.concatenate(out_meta, 0), np.concatenate(out_y, 0)


# =========================
# CORAL（二阶统计对齐）
# =========================
def _mat_power_symmetric(A: np.ndarray, power: float, eps: float = 1e-6) -> np.ndarray:
    """对称矩阵的幂: A^{power}，用特征分解"""
    w, V = np.linalg.eigh(A)
    w = np.maximum(w, eps)
    Wp = np.diag(w ** power)
    return (V @ Wp @ V.T).astype(np.float32)


def coral_fit_transform(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    对每个标签分布做 whiten->recolor 到所有样本的平均协方差。
    假设 X 已标准化零均值。
    """
    D = X.shape[1]
    eps = 1e-6
    labels = np.unique(y)
    cov_all = np.cov(X, rowvar=False) + eps * np.eye(D, dtype=np.float32)
    cov_all_sqrt = _mat_power_symmetric(cov_all, 0.5, eps)

    X_new = X.copy()
    for lb in labels:
        idx = np.where(y == lb)[0]
        if idx.size < 2:
            continue
        Xi = X[idx]
        cov_i = np.cov(Xi, rowvar=False) + eps * np.eye(D, dtype=np.float32)
        cov_i_inv_sqrt = _mat_power_symmetric(cov_i, -0.5, eps)
        A = (cov_i_inv_sqrt @ cov_all_sqrt).astype(np.float32)
        X_new[idx] = Xi @ A.T
    return X_new


# =========================
# t-SNE 与绘图
# =========================
def _auto_perplexity(perplexity: Optional[float], n_samples: int) -> float:
    if perplexity is None:
        p = 30.0
    else:
        p = float(perplexity)
    # t-SNE 要求 N > 3*perplexity
    ub = max(5.0, (n_samples - 1) / 3.0)
    if p > ub:
        print(f"[WARN] perplexity={p} 太大，自动裁剪为 {ub:.2f}（N={n_samples}）")
        p = max(5.0, min(p, ub))
    return p


def _parse_version(ver: str) -> Tuple[int, int, int]:
    """把 '1.3.2' -> (1,3,2)，不合法部分当 0。"""
    parts = (ver or "0").split(".")
    nums = []
    for i in range(3):
        try:
            nums.append(int(parts[i]))
        except Exception:
            nums.append(0)
    return tuple(nums)  # (major, minor, patch)


def run_tsne(
    X: np.ndarray,
    seed: int,
    perplexity: float,
    n_iter: int,
    metric: str = "euclidean",
    learning_rate: str = "auto",
    init: str = "pca",
) -> np.ndarray:
    """
    兼容不同 sklearn 版本的 t-SNE：
    - 仅传入构造器支持的参数（用 inspect.signature 过滤）
    - 老版本不支持 learning_rate='auto'，自动回退到 200.0
    - 某些实现不支持 n_iter，则使用默认迭代步数
    """
    from sklearn.manifold import TSNE  # 局部导入，确保拿到真正使用的 TSNE
    sig = inspect.signature(TSNE.__init__)
    accepts = set(sig.parameters.keys())

    # 版本判断：老版本（<1.2）没有 'auto' 学习率
    sk_major, sk_minor, sk_patch = _parse_version(getattr(sklearn, "__version__", "0"))
    allow_lr_auto = (sk_major, sk_minor) >= (1, 2)

    # 组装 kwargs（只放入构造函数确实接受的键）
    kwargs = {"n_components": 2}
    if "perplexity" in accepts:
        kwargs["perplexity"] = perplexity
    if "n_iter" in accepts:
        kwargs["n_iter"] = int(n_iter)
    else:
        print("[TSNE][warn] 该 sklearn 版本不支持 'n_iter'，将使用默认迭代步数。")
    if "metric" in accepts:
        kwargs["metric"] = metric
    if "learning_rate" in accepts:
        if isinstance(learning_rate, str):
            if allow_lr_auto and learning_rate == "auto":
                kwargs["learning_rate"] = "auto"
            else:
                # 老版本不接受 'auto'，回退到数值
                kwargs["learning_rate"] = 200.0
                if learning_rate == "auto":
                    print("[TSNE][warn] 当前 sklearn 不支持 learning_rate='auto'，已回退为 200.0")
        else:
            kwargs["learning_rate"] = float(learning_rate)
    if "init" in accepts:
        kwargs["init"] = init
    if "random_state" in accepts:
        kwargs["random_state"] = seed
    if "verbose" in accepts:
        kwargs["verbose"] = 1

    # 构造 + 拟合
    tsne = TSNE(**kwargs)
    Z = tsne.fit_transform(X)
    return Z.astype(np.float32)


def plot_scatter(
    Z: np.ndarray,
    y_str: np.ndarray,                  # 字符串标签
    labels_order: List[str],            # 按 runs 配置顺序
    label_to_color: Dict[str, str],
    out_path: str,                      # 输出 PDF 路径
    title: str,
    point_size: float = 6.0,
    alpha: float = 0.4,
    dpi: int = 160,
    legend_loc: str = "best",
):
    plt.figure(figsize=(7.2, 6.4), dpi=dpi)
    ax = plt.gca()
    for lb in labels_order:
        idx = np.where(y_str == lb)[0]
        if idx.size == 0:
            continue
        ax.scatter(
            Z[idx, 0], Z[idx, 1],
            s=point_size, alpha=alpha,
            c=label_to_color.get(lb, None),
            label=lb, edgecolors="none"
        )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    ax.legend(loc=legend_loc, frameon=True)
    ax.set_aspect("equal", adjustable="datalim")
    plt.tight_layout()
    # 保存为 PDF（矢量）
    plt.savefig(out_path)
    plt.close()
    print(f"[SAVE] {out_path}")


def save_embedding_csv_npy(
    Z: np.ndarray,
    y_str: np.ndarray,
    meta: np.ndarray,
    out_prefix: str,
):
    # CSV: x,y,label,episode,step,T
    try:
        import pandas as pd
        df = pd.DataFrame({
            "x": Z[:, 0],
            "y": Z[:, 1],
            "label": y_str,
            "episode": meta[:, 0],
            "step": meta[:, 1],
            "T": meta[:, 2],
        })
        csv_path = out_prefix + ".csv"
        df.to_csv(csv_path, index=False)
        print(f"[SAVE] {csv_path}")
    except Exception as e:
        print(f"[WARN] 保存 CSV 失败（缺少 pandas 或其他原因）：{e}")

    npy_path = out_prefix + ".npy"
    np.save(npy_path, Z)
    print(f"[SAVE] {npy_path}")


# =========================
# 主流程
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="YAML 配置文件路径")
    args = parser.parse_args()

    plot_cfg, runs_cfg = parse_config(args.config)
    os.makedirs(plot_cfg.out_dir, exist_ok=True)
    np.random.seed(plot_cfg.seed)

    # 颜色映射（按 runs 顺序）
    default_colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#e377c2", "#7f7f7f"]
    label_to_color = {}
    for i, rc in enumerate(runs_cfg):
        label_to_color[rc.name] = rc.color if rc.color else default_colors[i % len(default_colors)]

    # 读取所有 run 的数据
    sel_key = plot_cfg.key
    all_blocks: List[Tuple[np.ndarray, np.ndarray, str]] = []  # (X, meta, label_str)

    for rc in runs_cfg:
        tdir = _ensure_traj_dir(rc.traj_dir)
        npz_files = _list_npz(tdir)
        if not npz_files:
            print(f"[WARN] 目录无 npz：{tdir}")
            continue

        # 过滤 episode
        chosen = []
        for pth in npz_files:
            ep = _episode_id_from_npz(pth)
            if rc.include_episodes is not None and ep not in rc.include_episodes:
                continue
            chosen.append(pth)
        if rc.max_eps is not None:
            chosen = chosen[: rc.max_eps]

        print(f"[LOAD] {rc.name}: {len(chosen)} files from {tdir}")

        for npz_path in chosen:
            try:
                X, meta = _load_points_from_npz(npz_path, sel_key, plot_cfg.mode, plot_cfg.max_per_traj)
            except KeyError as e:
                print(f"[SKIP] {os.path.basename(npz_path)}: {e}")
                continue
            except Exception as e:
                print(f"[SKIP] {os.path.basename(npz_path)}: load failed -> {e}")
                continue

            if X.ndim != 2 or X.shape[0] == 0:
                continue
            all_blocks.append((X, meta, rc.name))

    if not all_blocks:
        raise RuntimeError("没有可用数据（检查 key、目录与过滤条件）")

    # 维度一致性 + 拼接
    X_all, meta_all, y_all = _concat_with_label(all_blocks, sel_key)

    # labels 顺序严格按 runs_cfg，但只保留实际出现的
    present = set([b[2] for b in all_blocks])
    labels_order = [rc.name for rc in runs_cfg if rc.name in present]

    # 平衡采样（按 label）
    if plot_cfg.balance_by_label and plot_cfg.max_samples_per_label is not None:
        X_all, meta_all, y_all = _balance_by_label(X_all, meta_all, y_all, plot_cfg.max_samples_per_label)

    N, D = X_all.shape
    print(f"[INFO] Total samples: {N}, dim={D}, labels={labels_order}")

    # 预处理：标准化（联合）
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_all).astype(np.float32)

    # 可选 CORAL（让不同标签的二阶统计对齐）
    if plot_cfg.coral and len(labels_order) > 1:
        print("[INFO] Apply CORAL...")
        Xs = coral_fit_transform(Xs, y_all)

    # 可选 PCA
    if plot_cfg.use_pca and D > plot_cfg.pca_dim:
        print(f"[INFO] PCA -> {plot_cfg.pca_dim}")
        pca = PCA(n_components=plot_cfg.pca_dim, random_state=plot_cfg.seed, svd_solver="auto")
        Xp = pca.fit_transform(Xs).astype(np.float32)
    else:
        Xp = Xs

    # t-SNE
    perplexity = _auto_perplexity(plot_cfg.perplexity, Xp.shape[0])
    print(f"[INFO] t-SNE: perplexity={perplexity}, n_iter={plot_cfg.n_iter}, metric={plot_cfg.metric}, init={plot_cfg.init}")
    Z = run_tsne(
        Xp,
        seed=plot_cfg.seed,
        perplexity=perplexity,
        n_iter=plot_cfg.n_iter,
        metric=plot_cfg.metric,
        learning_rate=plot_cfg.learning_rate,
        init=plot_cfg.init,
    )

    # 保存
    base = f"tsne_{plot_cfg.key}_{plot_cfg.mode}"
    out_pdf = os.path.join(plot_cfg.out_dir, base + ".pdf")   # <- PDF 矢量图
    out_prefix = os.path.join(plot_cfg.out_dir, base)

    label_to_color_ordered = {lb: label_to_color[lb] for lb in labels_order}
    plot_scatter(
        Z, y_all, labels_order, label_to_color_ordered,
        out_path=out_pdf,
        title=f"{plot_cfg.key} | {plot_cfg.mode} | N={N}",
        point_size=plot_cfg.point_size,
        alpha=plot_cfg.alpha,
        dpi=plot_cfg.dpi,
        legend_loc=plot_cfg.legend_loc,
    )

    try:
        save_embedding_csv_npy(Z, y_all, meta_all, out_prefix)
    except Exception as e:
        print(f"[WARN] 保存嵌入失败：{e}")

    print("[DONE]")


if __name__ == "__main__":
    main()