import argparse
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import onnx

def load_weights(model_path):
    model = onnx.load(model_path)
    weights = {init.name: onnx.numpy_helper.to_array(init) for init in model.graph.initializer}
    return weights

def visualize_side_by_side(kernels1:np.array, kernels2:np.array, save_path:Path):
    def assert_three_channels(weights):
        if len(weights.shape) != 4 or weights.shape[1] != 3: raise RuntimeError("Layer is not of expected structure (:,3,:,:).")

    def to_rgb(kernel):
        kernel = np.transpose(kernel, (1, 2, 0))  # CHW → HWC
        return (kernel - kernel.min()) / (kernel.max() - kernel.min())

    assert_three_channels(kernels1)
    assert_three_channels(kernels2)
    assert kernels1.shape == kernels2.shape
    num_filters = kernels1.shape[0]
    grid_cols = 16
    grid_rows = int(np.ceil(num_filters / grid_cols))
    fig, axes = plt.subplots(grid_rows * 2, grid_cols, figsize=(grid_cols * 1.8, grid_rows * 4.8))
    axes = np.array(axes).reshape(grid_rows * 2, grid_cols)
    for i in range(num_filters):
        row = i // grid_cols
        col = i % grid_cols
        fp_img = to_rgb(kernels1[i])
        q_img = to_rgb(kernels2[i])
        ax_fp = axes[row * 2][col]
        ax_q  = axes[row * 2 + 1][col]
        ax_fp.imshow(fp_img)
        ax_fp.set_title(f"FP32 #{i}", fontsize=20)
        ax_fp.axis("off")
        ax_q.imshow(q_img)
        ax_q.set_title(f"INT8 #{i}", fontsize=20)
        ax_q.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    fig.savefig(save_path, dpi=200)
    print(f"Saved visualization → {save_path}")

def visualize_kernels_pca(fp32_kernels: np.ndarray, deq_kernels: np.ndarray, save_path: Path):
    fp32_flat = fp32_kernels.reshape(fp32_kernels.shape[0], -1)
    deq_flat  = deq_kernels.reshape(deq_kernels.shape[0], -1)
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(np.vstack([fp32_flat, deq_flat]))
    fp32_pcs = pcs[:fp32_flat.shape[0]]
    deq_pcs  = pcs[fp32_flat.shape[0]:]
    plt.figure(figsize=(10, 10))
    plt.scatter(fp32_pcs[:, 0], fp32_pcs[:, 1], label="FP32", alpha=0.7)
    plt.scatter(deq_pcs[:, 0], deq_pcs[:, 1], label="Dequantized", alpha=0.7)
    plt.xlabel("PC1", fontsize=20)
    plt.ylabel("PC2", fontsize=20)
    plt.title("PCA of Convolution Kernels (FP32 vs Dequantized)", fontsize=24)
    plt.legend(fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def plot_mse(weights, save_path: Path):
    mse_values = []
    layer_names = []

    # Compute MSEs
    for name, _weights in weights.items():
        if len(_weights["fp32"].flatten()) < 1000: continue # skip bias terms
        if name in ["fc.bias", 'fc.weight']: continue
        layer_names.append(name)
        mse = float(np.mean((_weights["fp32"].flatten() - _weights["dequantized"].flatten()) ** 2))
        mse_values.append(mse)

    # Identify min/max layers
    min_idx = int(np.argmin(mse_values))
    max_idx = int(np.argmax(mse_values))
    print(f"Min MSE → Layer {min_idx+1}: {layer_names[min_idx]}  MSE={mse_values[min_idx]:.6e}")
    print(f"Max MSE → Layer {max_idx+1}: {layer_names[max_idx]}  MSE={mse_values[max_idx]:.6e}")

    # Plot
    x = np.arange(1, len(mse_values) + 1)
    plt.figure(figsize=(12, 6))
    plt.plot(x, mse_values, marker="o")
    plt.xlabel("Layer", fontsize=16)
    plt.ylabel("MSE", fontsize=16)
    plt.xticks(x, x, fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def plot_mse_and_outliers_severity(weights, save_path: Path):
    mse_values = []
    outlier_iqr_values = []
    layer_names = []

    # Compute MSEs + Outlier IQR severity
    for name, _weights in weights.items():
        if len(_weights["fp32"].flatten()) < 1000: 
            continue  # skip tiny layers (bias, BN, etc.)
        if name in ["fc.bias", "fc.weight"]:
            continue

        layer_names.append(name)

        fp32_vec = _weights["fp32"].flatten()
        deq_vec = _weights["dequantized"].flatten()

        mse = float(np.mean((fp32_vec - deq_vec) ** 2))
        mse_values.append(mse)

        # Outlier severity in IQR units
        q1 = np.percentile(fp32_vec, 25)
        q3 = np.percentile(fp32_vec, 75)
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr

        lower_outliers = fp32_vec[fp32_vec < lower_fence]
        upper_outliers = fp32_vec[fp32_vec > upper_fence]

        lower_outlier_iqr = float((lower_fence - lower_outliers.min()) / iqr) if len(lower_outliers) > 0 else 0.0
        upper_outlier_iqr = float((upper_outliers.max() - upper_fence) / iqr) if len(upper_outliers) > 0 else 0.0

        outlier_iqr_values.append(lower_outlier_iqr + upper_outlier_iqr)

    # Identify min/max layers
    min_idx = int(np.argmin(mse_values))
    max_idx = int(np.argmax(mse_values))
    print(f"Min MSE → Layer {min_idx+1}: {layer_names[min_idx]}  MSE={mse_values[min_idx]:.6e}")
    print(f"Max MSE → Layer {max_idx+1}: {layer_names[max_idx]}  MSE={mse_values[max_idx]:.6e}")
    for i in range(len(layer_names)):
        print(f"({i+1}) {layer_names[i]}")

    # Plot with two y-axes
    x = np.arange(1, len(mse_values) + 1)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Left axis → MSE
    ax1.plot(x, mse_values, marker="o", color="blue", label="MSE")
    ax1.set_xlabel("Layer", fontsize=16)
    ax1.set_ylabel("MSE", fontsize=16, color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.set_xticks(x)
    ax1.set_xticklabels(x, fontsize=12)

    # Right axis → Outlier IQR severity
    ax2 = ax1.twinx()
    ax2.plot(x, outlier_iqr_values, marker="x", color="red", label="Outlier IQR Range")
    ax2.set_ylabel("Outliers range (IQR units)", fontsize=16, color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    # Grid + layout
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close()

@dataclass
class StatsSummary:
    count: int
    mean: float
    median: float
    std: float
    var: float
    min: float
    max: float
    q1: float
    q3: float
    iqr: float
    lower_outlier_iqr: float
    upper_outlier_iqr: float

class StatisticalAnalysis:
    def __init__(self, vector, results_dir: Path, run_name: str):
        self.vector = np.array(vector, dtype=float).flatten()
        self.stats = self._compute_stats()
        print(self.stats)
        self.name = run_name
        self.results_dir = results_dir
        self.plot_boxplot()
        self.plot_histogram()

    def _compute_stats(self):
        q1 = np.percentile(self.vector, 25)
        q3 = np.percentile(self.vector, 75)
        iqr = q3 - q1

        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr

        # Compute outlier distances in IQR units
        lower_outliers = self.vector[self.vector < lower_fence]
        upper_outliers = self.vector[self.vector > upper_fence]

        lower_outlier_iqr = float((lower_fence - lower_outliers.min()) / iqr) if len(lower_outliers) > 0 else 0.0
        upper_outlier_iqr = float((upper_outliers.max() - upper_fence) / iqr) if len(upper_outliers) > 0 else 0.0

        return StatsSummary(
            count=len(self.vector),
            mean=float(np.mean(self.vector)),
            median=float(np.median(self.vector)),
            std=float(np.std(self.vector, ddof=1)),
            var=float(np.var(self.vector, ddof=1)),
            min=float(np.min(self.vector)),
            max=float(np.max(self.vector)),
            q1=float(q1),
            q3=float(q3),
            iqr=float(iqr),
            lower_outlier_iqr=lower_outlier_iqr,
            upper_outlier_iqr=upper_outlier_iqr
        )

    def plot_boxplot(self):
        plt.figure(figsize=(4, 4))
        plt.boxplot(self.vector, vert=True, patch_artist=True)
        plt.title(f"Outlier Box Plot ({self.name})")
        plt.ylabel("Value")
        plt.grid(True, alpha=0.4)
        plt.margins(x=0.1)
        plt.gca().set_xmargin(0.05)
        plt.gca().autoscale(enable=True, tight=True)

        plt.tight_layout()
        plt.savefig(str(self.results_dir / f"{self.name}_boxplot.png"),
                    dpi=200,
                    bbox_inches="tight")
        plt.close()


    def plot_histogram(self):
        plt.figure(figsize=(8, 5))
        plt.hist(self.vector, bins='auto', alpha=0.7)
        plt.axvline(self.stats.mean, color='red', linestyle='--', label=f"Mean = {self.stats.mean:.2f}")
        plt.axvline(self.stats.median, color='green', linestyle='--', label=f"Median = {self.stats.median:.2f}")
        plt.axvline(self.stats.q1, color='orange', linestyle=':', label=f"Q1 = {self.stats.q1:.2f}")
        plt.axvline(self.stats.q3, color='orange', linestyle=':', label=f"Q3 = {self.stats.q3:.2f}")
        plt.title("Histogram with Statistical Properties")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True, alpha=0.4)
        plt.tight_layout()
        plt.savefig(str(self.results_dir / f"{self.name}_histogram.png"), dpi=200, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Path to FP32 ONNX model")
    parser.add_argument("quant_model_path", help="Path to INT8 ONNX model")
    args = parser.parse_args()
    fp32_weights = load_weights(args.model_path)
    int8_weights = load_weights(args.quant_model_path)

    joint_weights = {}
    for name in fp32_weights.keys():
        int8w = int8_weights[f"{name}_quantized"]
        scale = int8_weights[f"{name}_scale"] if f"{name}_scale" in int8_weights.keys() else int8_weights[f"{name}_quantized_scale"]
        zerop = int8_weights[f"{name}_zero_point"] if f"{name}_zero_point" in int8_weights.keys() else int8_weights[f"{name}_quantized_zero_point"]
        joint_weights[name] = { "fp32": fp32_weights[name], "quantized": int8w, "dequantized": (int8w - zerop) * scale }

    kernel_viz = joint_weights["onnx::Conv_193"]
    visualize_side_by_side(kernel_viz["fp32"], kernel_viz["dequantized"], Path("compare_fp32_int8_filters.png"))
    visualize_kernels_pca(kernel_viz["fp32"], kernel_viz["dequantized"], Path("pca.png"))
    plot_mse(joint_weights, "mse.png")

    res_dir = Path("statistics_plots")
    res_dir.mkdir(exist_ok=True, parents=True)
    stats_viz = joint_weights["onnx::Conv_238"]
    StatisticalAnalysis(stats_viz["fp32"], res_dir, "layer_16")
    StatisticalAnalysis(stats_viz["dequantized"], res_dir, "layer_16_q")
    res_dir.mkdir(exist_ok=True, parents=True)
    stats_viz = joint_weights["onnx::Conv_205"]
    StatisticalAnalysis(stats_viz["fp32"], res_dir, "layer_5")
    StatisticalAnalysis(stats_viz["dequantized"], res_dir, "layer_5_q")
    plot_mse_and_outliers_severity(joint_weights, "comp.png")
    stats_viz = joint_weights["onnx::Conv_211"]
    StatisticalAnalysis(stats_viz["fp32"], res_dir, "layer_7")
    StatisticalAnalysis(stats_viz["dequantized"], res_dir, "layer_7_q")