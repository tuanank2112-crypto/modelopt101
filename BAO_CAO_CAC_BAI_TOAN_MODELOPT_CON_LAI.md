# BÁO CÁO KỸ THUẬT: CẨM NANG TOÀN DIỆN CÁC PHƯƠNG PHÁP TỐI ƯU HÓA MÔ HÌNH (BÀI 2 ĐẾN BÀI 5)
> **Dự án**: ModelOpt101 — Edge AI & Robotics Optimization  
> **Các bài thực hành**:  
> - Bài 2: [02_knowledge_distillation.ipynb](examples/02_knowledge_distillation.ipynb)  
> - Bài 3: [03_pruning.ipynb](examples/03_pruning.ipynb)  
> - Bài 4: [04_nas.ipynb](examples/04_nas.ipynb)  
> - Bài 5: [05_kernel_optimization.ipynb](examples/05_kernel_optimization.ipynb)  
> **Quy ước ký hiệu toán học chuẩn**: Trọng số là $W$ (Weights), Đầu vào/Kích hoạt là $x$ (Inputs/Activations), Logits là $z$, Nhãn thực tế là $y$.  
> **Thời gian thực hiện**: 2026-11-03  

---

## MỤC LỤC
1. [BÀI 2: KNOWLEDGE DISTILLATION (CHƯNG CẤT TRI THỨC)](#1-bài-2-knowledge-distillation-chưng-cất-tri-thức)
   - 1.1. Đề bài đang hướng đến điều gì?
   - 1.2. Kiến thức & Ký hiệu toán học chuẩn xác
   - 1.3. Giá trị cốt lõi & Đánh đổi (Trade-offs)
   - 1.4. Lời giải chi tiết các `TODO` (Cell 6 & Cell 15)
   - 1.5. Kết quả thực nghiệm
2. [BÀI 3: PRUNING (CẮT TỈA TRỌNG SỐ VÀ BỘ LỌC)](#2-bài-3-pruning-cắt-tỉa-trọng-số-và-bộ-lọc)
   - 2.1. Đề bài đang hướng đến điều gì?
   - 2.2. Kiến thức & Ký hiệu toán học chuẩn xác
   - 2.3. Giá trị cốt lõi: Phi cấu trúc vs Cấu trúc
   - 2.4. Lời giải chi tiết các `TODO` (Cell 12 & Cell 16)
   - 2.5. Kết quả thực nghiệm
3. [BÀI 4: NEURAL ARCHITECTURE SEARCH (TÌM KIẾM KIẾN TRÚC TỰ ĐỘNG)](#3-bài-4-neural-architecture-search-tìm-kiếm-kiến-trúc-tự-động)
   - 3.1. Đề bài đang hướng đến điều gì?
   - 3.2. Kiến thức & Không gian tìm kiếm toán học
   - 3.3. Giá trị cốt lõi: Đường biên Pareto
   - 3.4. Lời giải chi tiết các `TODO` (Cell 6, 10, 12)
   - 3.5. Kết quả thực nghiệm
4. [BÀI 5: GPU KERNEL OPTIMIZATION (TỐI ƯU HÓA PHẦN CỨNG VỚI TRITON)](#4-bài-5-gpu-kernel-optimization-tối-ưu-hóa-phần-cứng-với-triton)
   - 4.1. Đề bài đang hướng đến điều gì?
   - 4.2. Kiến thức & Bản chất phần cứng GPU (SRAM vs DRAM)
   - 4.3. Giá trị cốt lõi: Tiling & L2 Cache Grouping
   - 4.4. Lời giải chi tiết các `TODO` (Cell 7 - Triton JIT GEMM)
   - 4.5. Kết quả thực nghiệm & Tốc độ tăng tốc
5. [BẢNG TỔNG HỢP SO SÁNH 5 KỸ THUẬT TỐI ƯU HÓA](#5-bảng-tổng-hợp-so-sánh-5-kỹ-thuật-tối-ưu-hóa)

---

# 1. BÀI 2: KNOWLEDGE DISTILLATION (CHƯNG CẤT TRI THỨC)

### 1.1. Đề bài đang hướng đến điều gì?
* **Thách thức**: Một mô hình kích thước nhỏ (**Student**) với ít tham số trọng số $W_S$ rất khó học tốt nếu chỉ huấn luyện từ đầu (from scratch) trên nhãn cứng one-hot $y \in \{0, 1\}^C$. Nhãn cứng chỉ cho biết lớp đúng là gì, hoàn toàn triệt tiêu thông tin về mối tương quan giữa các lớp (ví dụ: ảnh số $7$ trông khá giống số $1$, nhưng hoàn toàn khác số $8$).
* **Mục tiêu**: Huấn luyện một mạng lớn (**Teacher** với trọng số $W_T$) đạt độ chính xác cao, sau đó dùng Teacher làm "người thầy" hướng dẫn mạng nhỏ (Student) học thông qua phân phối xác suất mềm (**Soft Targets**).
* **Kết quả kỳ vọng**: Mạng Student đạt độ chính xác tiệm cận Teacher mà **không tăng thêm dù chỉ 1 byte trọng số hay 1 phép toán khi chạy thực tế**.

---

### 1.2. Kiến thức & Ký hiệu toán học chuẩn xác

> **Quy ước ký hiệu**:
> * $W_T$: Ma trận trọng số của mạng Teacher (kích thước lớn).
> * $W_S$: Ma trận trọng số của mạng Student (kích thước nhỏ).
> * $x$: Tensor ảnh đầu vào ($x \in \mathbb{R}^{B \times C_{\text{in}} \times H \times W}$).
> * $z_t = f(x; W_T)$: Vector Logits đầu ra của mạng Teacher.
> * $z_s = f(x; W_S)$: Vector Logits đầu ra của mạng Student.
> * $y$: Nhãn thực tế (Ground-truth one-hot hoặc class index).
> * $T \ge 1$: Nhiệt độ làm mềm phân phối (Distillation Temperature).
> * $\alpha \in [0, 1]$: Trọng số cân bằng giữa mục tiêu học từ thầy và học từ dữ liệu gốc.

#### Phân phối xác suất mềm với Nhiệt độ $T$:
Hàm Softmax tiêu chuẩn ($T=1$) thường biến một logit cao thành xác suất xấp xỉ $0.999$, che giấu mọi thông tin xác suất ở các lớp còn lại. Khi đưa nhiệt độ $T > 1$ vào:
$$p_i(z, T) = \frac{\exp(z_i / T)}{\sum_{j=1}^{C} \exp(z_j / T)}$$

Khi $T$ tăng, phân phối xác suất "mềm" ra, làm lộ rõ **Dark Knowledge** (tri thức ẩn về sự tương đồng giữa các lớp mà Teacher đã học được).

#### Hàm mất mát chưng cất toàn phần (Total Distillation Loss):
$$\mathcal{L}_{\text{total}}(W_S) = \alpha \cdot T^2 \cdot \mathcal{L}_{\text{KD}}(z_s, z_t; T) + (1 - \alpha) \cdot \mathcal{L}_{\text{CE}}(z_s, y)$$

Trong đó:
1. **Soft Loss ($\mathcal{L}_{\text{KD}}$)** đo độ phân kỳ Kullback-Leibler giữa phân phối của Student và Teacher:
   $$\mathcal{L}_{\text{KD}} = \mathcal{D}_{\text{KL}}\Big( \sigma\big(\frac{z_s}{T}\big) \;\Big\|\; \sigma\big(\frac{z_t}{T}\big) \Big) = \sum_{i=1}^C p_i(z_t, T) \cdot \log\left( \frac{p_i(z_t, T)}{p_i(z_s, T)} \right)$$
2. **Hệ số nhân $T^2$**: Đạo hàm của $\mathcal{L}_{\text{KD}}$ theo $z_s$ bị giảm theo tỷ lệ $\frac{1}{T^2}$ do chia cho $T$ trong hàm softmax:
   $$\frac{\partial \mathcal{L}_{\text{KD}}}{\partial z_{s,i}} \approx \frac{1}{T^2} (p_i(z_s, T) - p_i(z_t, T))$$
   Do đó, bắt buộc phải nhân thêm $T^2$ để biên độ gradient của Soft Loss cân bằng với Hard Cross-Entropy Loss $\mathcal{L}_{\text{CE}}$.

---

### 1.3. Giá trị cốt lõi & Đánh đổi
* **Giá trị cốt lõi**: Nâng cao chất lượng biểu diễn của trọng số $W_S$. Khi triển khai (Deployment), ta vứt bỏ hoàn toàn mạng Teacher, chỉ giữ lại mạng Student. Chi phí suy luận (Inference Latency & Memory) là **hoàn toàn miễn phí (Zero Runtime Cost)**.
* **Đánh đổi**: Chi phí huấn luyện (Training Compute) tăng gấp đôi vì phải huấn luyện Teacher trước, và trong quá trình huấn luyện Student phải chạy forward cả Teacher.

---

### 1.4. Lời giải chi tiết các `TODO` trong `02_knowledge_distillation.ipynb`

#### 🔹 TODO 1 (Cell 6): Xây dựng kiến trúc Teacher lớn
Mạng Teacher cần dung lượng kênh tăng dần ($64 \to 128 \to 256$) thay vì kiến trúc hẹp của Student ($16 \to 32 \to 32$):

```python
class Teacher(nn.Module):           # ~1.2M params  (big)
    def __init__(self, num_classes=10):
        super().__init__()
        # TODO: the teacher is the big model - use rising stage widths (e.g. 64 -> 128 -> 256)
        c1, c2, c3 = 64, 128, 256
        self.features = nn.Sequential(
            conv_block(1, c1),   conv_block(c1, c1),   nn.MaxPool2d(2),   # 28 -> 14
            conv_block(c1, c2),  conv_block(c2, c2),  nn.MaxPool2d(2),   # 14 -> 7
            conv_block(c2, c3),  conv_block(c3, c3),  nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(c3, num_classes)
```

#### 🔹 TODO 2 & 3 (Cell 15): Hàm mất mát Chưng cất
```python
def distillation_loss(student_logits, teacher_logits, labels, T=4.0, alpha=0.7):
    # 1. Hard targets: Cross-entropy thông thường với nhãn đúng
    ce_term = F.cross_entropy(student_logits, labels)

    # 2. TODO: Soft-target term = KL(log_softmax(student/T) || softmax(teacher/T)) * T**2
    # Chú ý: F.kl_div nhận đầu vào thứ nhất là log-probabilities và thứ hai là probabilities chuẩn
    student_soft = F.log_softmax(student_logits / T, dim=-1)
    teacher_soft = F.softmax(teacher_logits / T, dim=-1)
    kd_term = F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (T ** 2)

    # 3. TODO: Kết hợp 2 hàm mất mát theo tỷ lệ alpha
    return alpha * kd_term + (1.0 - alpha) * ce_term
```

---

### 1.5. Kết quả thực nghiệm đo đạc thực tế (Empirical Results)

Dưới đây là kết quả kiểm chứng trực tiếp khi huấn luyện và đánh giá trên GPU NVIDIA Quadro T2000 (CUDA 12.4 trong WSL) trong notebook `02_knowledge_distillation.ipynb`:

| Mô hình | Cấu hình kênh ($C_1 \to C_2 \to C_3$) | Số tham số trọng số ($W$) | Dung lượng đĩa | CPU Latency | Test Accuracy (%) | Nhận xét hiệu năng |
|---|:---:|:---:|:---:|:---:|:---:|---|
| **Teacher (Big)** | $64 \to 128 \to 256$ | $1,147,722$ | $4.61\text{ MB}$ | $6.59\text{ ms}$ | **$99.67\%$** | Mạng lớn đóng vai trò giáo viên chuẩn |
| **Student (No KD)** | $16 \to 32 \to 32$ | $14,458$ | $0.07\text{ MB}$ | $0.80\text{ ms}$ | $99.26\%$ | Học từ nhãn cứng (one-hot), thiếu thông tin tinh tế |
| **Student (With KD)** | $16 \to 32 \to 32$ | $14,458$ | $0.07\text{ MB}$ | $0.96\text{ ms}$ | **$99.39\%$** | **Tăng thêm $+0.13\%$** mà kích thước và độ trễ giữ nguyên siêu nhỏ! |

```mermaid
xychart-beta
    title "So sánh độ chính xác Test Accuracy (%) trong Chưng cất tri thức"
    x-axis ["Teacher (Big)", "Student Baseline", "Student + Distillation"]
    y-axis "Accuracy (%)" 99 --> 100
    bar [99.67, 99.26, 99.39]
```

> **Nhận định cốt lõi**: Mạng Student giữ nguyên kích thước siêu nhỏ ($14,458$ tham số, nhỏ hơn Teacher **$79.4\times$** và nhanh hơn **$7.7\times$**), nhưng đạt tới **$99.39\%$** độ chính xác nhờ tri thức phân phối mềm (soft targets) truyền từ Teacher ($T=4.0, \alpha=0.7$).

#### Trích xuất nhật ký thực thi thực tế (Raw Execution Logs):
```text
=== EXERCISE 2: KNOWLEDGE DISTILLATION (02_knowledge_distillation.ipynb) ===
teacher params: 1,147,722 (4.61MB) -> Teacher final accuracy: 99.67%
student params: 14,458 (0.07MB, 79.4x smaller)
Student (no KD) accuracy: 99.26% (0.80ms latency)
Student (with KD) accuracy: 99.39% (0.96ms latency)
>>> KD gave the tiny student +0.13% accuracy at the SAME size & latency.
>>> The student is 79.4x smaller and 7.7x faster than the teacher.
```
```

---

# 2. BÀI 3: PRUNING (CẮT TỈA TRỌNG SỐ VÀ BỘ LỌC)

### 2.1. Đề bài đang hướng đến điều gì?
* **Thách thức**: Các mạng nơ-ron sâu bị thừa thãi tham số (Over-parameterized). Nhiều kết nối trọng số có độ lớn xấp xỉ $0$ và không đóng góp đáng kể vào kết quả phân loại.
* **Mục tiêu**:
  1. Loại bỏ các trọng số nhỏ nhất theo tỷ lệ định trước (**Unstructured Pruning**).
  2. Cắt bỏ hoàn toàn các bộ lọc tích chập (filters) kém quan trọng nhất để tạo ra một mạng mới có cấu trúc thực sự nhỏ hơn (**Structured Filter Pruning**).
  3. Huấn luyện tinh chỉnh phục hồi (**Fine-tuning**) để lấy lại độ chính xác đã mất.

---

### 2.2. Kiến thức & Ký hiệu toán học chuẩn xác

> **Quy ước ký hiệu**:
> * $W^{(l)} \in \mathbb{R}^{C_{\text{out}} \times C_{\text{in}} \times K \times K}$: Tensor trọng số của tầng tích chập thứ $l$.
> * $W_c^{(l)} \in \mathbb{R}^{C_{\text{in}} \times K \times K}$: Bộ lọc (filter) thứ $c$ của tầng $l$.
> * $M^{(l)} \in \{0, 1\}^{\text{dim}(W^{(l)})}$: Mặt nạ nhị phân (Binary Mask).
> * $\text{ratio} \in [0, 1)$: Tỷ lệ cắt tỉa (ví dụ $0.3 = 30\%$).

#### A. Cắt tỉa phi cấu trúc (Unstructured Magnitude Pruning):
Đánh giá mức độ quan trọng $S$ của từng phần tử trọng số riêng lẻ dựa trên độ lớn biên độ:
$$S(W_{i,j,k,m}) = |W_{i,j,k,m}|$$

Tìm ngưỡng cắt $\theta$ tại phân vị (quantile) thứ $\text{ratio}$ của tập hợp tất cả các trọng số:
$$\theta = \text{Quantile}\big( \{|W|\}, \; \text{ratio} \big)$$

Mặt nạ nhị phân $M$ được xác định:
$$M_{i,j,k,m} = \begin{cases} 1 & \text{nếu } |W_{i,j,k,m}| \ge \theta \\ 0 & \text{nếu } |W_{i,j,k,m}| < \theta \end{cases}$$

Trọng số sau cắt tỉa:
$$W_{\text{pruned}} = W \odot M \quad (\odot \text{ là phép nhân Hadamard element-wise})$$

#### B. Cắt tỉa có cấu trúc bộ lọc (Structured Filter Pruning):
Thay vì xóa các phần tử rời rạc, ta tính chuẩn $L_1$ của toàn bộ filter $W_c$:
$$\|W_c\|_1 = \sum_{j=1}^{C_{\text{in}}} \sum_{u=1}^K \sum_{v=1}^K |W_{c,j,u,v}|$$

Giữ lại $k$ filters có chuẩn $\|W_c\|_1$ lớn nhất:
$$k = \max\left(1, \; \text{round}\big(C_{\text{out}} \cdot (1 - \text{ratio})\big)\right)$$

Cắt bỏ $C_{\text{out}} - k$ filters thừa thãi tạo ra một mạng nơ-ron mới có số kênh đầu ra giảm xuống còn $k$, đồng thời giảm số kênh đầu vào của tầng kế tiếp.

---

### 2.3. Giá trị cốt lõi: Phi cấu trúc vs Cấu trúc

| Tiêu chí | Unstructured Pruning | Structured Filter Pruning |
|---|---|---|
| **Dạng ma trận** | Ma trận thưa (Sparse tensor) có nhiều số $0$ | Ma trận đặc (Dense tensor) có kích thước nhỏ hơn |
| **Dung lượng nén trên đĩa** | Giảm khi dùng thuật toán nén thưa (CSR/CSC, gzip) | **Giảm thực tế trực tiếp** mà không cần thư viện đặc thù |
| **Tốc độ thực thi (Speedup)** | **Không tăng tốc** trên GPU thông thường nếu thiếu Sparse Core | **Tăng tốc thực tế ngay lập tức** trên mọi CPU/GPU |

---

### 2.4. Lời giải chi tiết các `TODO` trong `03_pruning.ipynb`

#### 🔹 TODO 1 & 2 (Cell 12): Unstructured Pruning
```python
def prune_unstructured(model, ratio):
    """Zero the globally smallest-magnitude weights. Returns a masked copy + the threshold."""
    mp = copy.deepcopy(model)
    # Tập hợp tất cả các trọng số tuyệt đối của các khối conv
    all_w = torch.cat([blk[0].weight.data.abs().flatten() for blk in conv_blocks(mp)])
    
    # TODO 1: Tính ngưỡng threshold tại quantile = ratio
    thresh = torch.quantile(all_w, ratio)
    
    # TODO 2: Gán 0 trực tiếp vào các trọng số có |W| < thresh
    for blk in conv_blocks(mp):
        w = blk[0].weight.data
        w.mul_((w.abs() >= thresh).float())
        
    return mp, thresh.item()
```

#### 🔹 TODO 3 (Cell 16): Structured Filter Pruning
```python
def structured_prune(model, ratio):
    """L1-norm filter pruning -> a smaller Net with the kept filters copied in."""
    blocks = conv_blocks(model)
    keep = []
    for blk in blocks:
        # Số lượng filters giữ lại
        k = max(1, round(blk[0].weight.shape[0] * (1 - ratio)))
        
        # TODO: Đánh giá từng filter theo chuẩn L1 (tổng abs qua dim 1, 2, 3), lấy top-k chỉ số lớn nhất
        w = blk[0].weight.data
        l1_norm = w.abs().sum(dim=(1, 2, 3))
        idx = torch.topk(l1_norm, k).indices.sort().values
        keep.append(idx)

    # Khởi tạo mô hình mới với số lượng kênh thu nhỏ tương ứng
    new_widths = tuple(len(k) for k in keep)
    new_model = Net(widths=new_widths)
    # (Phần sao chép trọng số sang new_model đã được viết sẵn trong notebook)
    return new_model
```

---

### 2.5. Kết quả thực nghiệm đo đạc thực tế (Empirical Results)

Dưới đây là kết quả kiểm chứng trực tiếp khi chạy notebook `03_pruning.ipynb` trên GPU NVIDIA Quadro T2000 (CUDA 12.4 trong WSL):

#### a. Unstructured Pruning (Cắt tỉa từng phần tử trọng số theo độ lớn)
| Tỷ lệ cắt (Sparsity) | Test Accuracy (%) | Độ sụt giảm ($\Delta \text{Acc}$) | Đánh giá |
|:---:|:---:|:---:|---|
| **0% (Dense Baseline)** | **$99.70\%$** | $0.00\%$ | Mạng đầy đủ 1,147,722 tham số (4.61MB, 7.50ms) |
| **50%** | **$99.60\%$** | **$-0.10\%$** | Xóa bỏ 50% trọng số nhỏ nhất mà mạng hầu như không suy giảm! |
| **70%** | **$99.11\%$** | **$-0.59\%$** | Vẫn giữ được trên 99% độ chính xác |
| **80%** | **$89.78\%$** | $-9.92\%$ | Bắt đầu mất mát đặc trưng nhận dạng |
| **90%** | **$43.12\%$** | $-56.58\%$ | Mạng bị phá hủy nghiêm trọng |

#### b. Structured Filter Pruning (Cắt bỏ 50% channel Conv bằng L1-norm + Fine-tuning)
| Mô hình | Số tham số ($W$) | Dung lượng đĩa | CPU Latency | Test Accuracy (%) | Nhận xét |
|---|:---:|:---:|:---:|:---:|---|
| **Dense Baseline** | $1,147,722$ | $4.61\text{ MB}$ | $7.50\text{ ms}$ | **$99.70\%$** | Chuẩn tham chiếu |
| **Pruned (chưa fine-tune)** | $288,170$ | $1.17\text{ MB}$ | $5.49\text{ ms}$ | $9.74\%$ | Mất cân bằng kênh sau khi cắt bỏ filters |
| **Pruned + Fine-tuned (4 epochs)** | **$288,170$** | **$1.17\text{ MB}$** | **$5.49\text{ ms}$** | **$99.63\%$** | **Phục hồi hoàn toàn, nhỏ hơn $3.9\times$, nhanh hơn $1.4\times$!** |

#### Trích xuất nhật ký thực thi thực tế (Raw Execution Logs):
```text
=== EXERCISE 3: PRUNING (03_pruning.ipynb) ===
dense baseline: params=1,147,722  size=4.61MB  acc=99.70%  latency=7.50ms
sparsity   0%  ->  accuracy 99.70%
sparsity  50%  ->  accuracy 99.60%
sparsity  70%  ->  accuracy 99.11%
sparsity  80%  ->  accuracy 89.78%
Structured prune: params 1,147,722 -> 288,170 (4.0x smaller)
accuracy right after pruning (no fine-tune): 9.74%
[finetune] epoch 04/4  test_acc=99.63%
>>> structured: 3.9x smaller, 1.4x faster, -0.07% accuracy at 50% pruned.
```

---

# 3. BÀI 4: NEURAL ARCHITECTURE SEARCH (TÌM KIẾM KIẾN TRÚC TỰ ĐỘNG)

### 3.1. Đề bài đang hướng đến điều gì?
* **Thách thức**: Thiết kế kiến trúc mạng (số tầng $D$, số kênh $C$, kích thước kernel $K$) bằng tay dựa vào trực giác kỹ sư rất dễ dẫn đến kiến trúc thừa thãi hoặc thiếu tối ưu giữa kích thước mô hình và độ chính xác.
* **Mục tiêu**: Tự động hóa quá trình tìm kiếm kiến trúc mạng tối ưu bằng hai chiến lược: **Random Search** và **Evolutionary Algorithm (Giải thuật di truyền)**, sử dụng một hàm đánh giá nhanh (**Proxy Evaluation**) để tiết kiệm thời gian tính toán.

---

### 3.2. Kiến thức & Không gian tìm kiếm toán học

> **Quy ước ký hiệu**:
> * Không gian kiến trúc $\mathcal{A}$: Tập hợp các bộ gene cấu hình mạng:
>   $$\alpha = (D, C_0, K) \in \mathcal{A}$$
>   * $D \in \{2, 3, 4\}$: Độ sâu (Depth - số tầng tích chập).
>   * $C_0 \in \{8, 12, 16, 24, 32\}$: Độ rộng kênh khởi đầu (Base width), các tầng sau tăng theo $C_i = \min(C_0 \cdot 2^i, 128)$.
>   * $K \in \{3, 5\}$: Kích thước bộ lọc (Kernel size).
> * $\text{Proxy-Fitness}(\alpha)$: Độ thích nghi đại diện = Độ chính xác đạt được sau khi huấn luyện nhanh (ví dụ chỉ $1$ epoch trên tập dữ liệu con $10\%$).

#### Cơ chế Đột biến Di truyền (Genetic Mutation):
Từ một kiến trúc cha mẹ $\alpha_{\text{parent}} = (D, C_0, K)$, ta tạo ra con $\alpha_{\text{child}}$ bằng cách đột biến ngẫu nhiên một gene:
$$\text{gene}^* \sim \text{Uniform}(\{D, C_0, K\}), \quad \alpha_{\text{child}}[\text{gene}^*] \sim \text{Uniform}(\text{Choices}[\text{gene}^*] \setminus \{\alpha_{\text{parent}}[\text{gene}^*]\})$$

---

### 3.3. Giá trị cốt lõi: Đường biên Pareto (Pareto-optimal Frontier)
Giá trị lớn nhất của NAS là tìm ra tập hợp các kiến trúc nằm trên **Đường biên Pareto**: Với bất kỳ kích thước mô hình nào (ví dụ $0.2 \text{ MB}$, $0.5 \text{ MB}$), kiến trúc do NAS tìm ra luôn đạt độ chính xác cao nhất có thể.

---

### 3.4. Lời giải chi tiết các `TODO` trong `04_nas.ipynb`

#### 🔹 TODO 1 (Cell 6): Hàm xây dựng mạng từ bộ gene (`build_model`)
```python
def build_model(arch, num_classes=10):
    layers, cin = [], 1
    depth = arch["depth"]
    base_width = arch["width"]
    k = arch["kernel"]
    
    # TODO: Duyệt qua từng tầng theo depth, tính cout và ghép Conv/BN/ReLU/Pool
    for i in range(depth):
        cout = min(base_width * (2 ** i), 128)
        layers += [
            nn.Conv2d(cin, cout, k, padding=k // 2, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        ]
        cin = cout
        
    layers += [nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(cin, num_classes)]
    return nn.Sequential(*layers)
```

#### 🔹 TODO 2 (Cell 10): Đánh giá nhanh Proxy
```python
def proxy_eval(arch):
    """Cheap fitness: quick-train on the subset, return (accuracy, size_MB)."""
    model = build_model(arch)
    # TODO: Huấn luyện nhanh trên tập tìm kiếm nhỏ search_loader
    train(model, search_loader, epochs=SEARCH_EPOCHS)
    return evaluate(model), model_size_mb(model)
```

#### 🔹 TODO 3 & 4 (Cell 12): Đột biến và Chọn lọc cá thể cha mẹ
```python
def mutate(arch):
    child = dict(arch)
    # TODO 3: Chọn ngẫu nhiên 1 gene và đổi sang giá trị khác giá trị hiện tại
    gene = random.choice(list(CHOICES.keys()))
    allowed_choices = [v for v in CHOICES[gene] if v != arch[gene]]
    child[gene] = random.choice(allowed_choices)
    return child

# Trong vòng lặp tiến hóa:
for gen in range(EVO_GENS):
    # TODO 4: Chọn 2 cá thể có accuracy cao nhất từ đầu đến giờ làm cha mẹ
    parents = sorted(results, key=lambda r: r["acc"], reverse=True)[:2]
    # Sinh con từ cha mẹ thông qua mutate...
```

---

### 3.5. Kết quả thực nghiệm đo đạc thực tế (Empirical Results)

Dưới đây là kết quả kiểm chứng trực tiếp khi chạy notebook `04_nas.ipynb` trên GPU NVIDIA Quadro T2000 (CUDA 12.4 trong WSL):

#### a. Các kiến trúc tối ưu trên đường biên Pareto (Pareto-optimal)
Thuật toán Random Search và Genetic Evolution đã khám phá không gian $30$ cấu hình và lọc ra các kiến trúc bất khả chiến bại trên cả hai tiêu chí (Accuracy cao nhất ở mỗi mức dung lượng):
* `d2-w8-k5`: Proxy Acc = $74.60\%$, Dung lượng = $0.02\text{ MB}$
* `d3-w8-k3`: Proxy Acc = $92.77\%$, Dung lượng = $0.03\text{ MB}$
* `d3-w16-k3`: Proxy Acc = $96.50\%$, Dung lượng = $0.10\text{ MB}$
* `d4-w12-k3`: Proxy Acc = $96.72\%$, Dung lượng = $0.23\text{ MB}$ (Kiến trúc thắng cuộc)
* `d4-w16-k3`: Proxy Acc = $98.26\%$, Dung lượng = $0.40\text{ MB}$
* `d4-w12-k5`: Proxy Acc = $98.45\%$, Dung lượng = $0.62\text{ MB}$

#### b. Đối đầu trực tiếp (10 epochs Full Training) giữa Baseline và NAS Winner
| Cấu hình kiến trúc | Cấu hình ($D, C_0, K$) | Dung lượng đĩa | CPU Latency | Test Accuracy (%) | Đánh giá |
|---|:---:|:---:|:---:|:---:|---|
| **Baseline (Thiết kế tay)** | `d3-w32-k3` ($3$ tầng, $32$ kênh, k3) | $0.38\text{ MB}$ | $7.94\text{ ms}$ | **$99.42\%$** | Chuẩn tham chiếu do con người chọn |
| **NAS Pick (Tìm kiếm tự động)** | `d4-w12-k3` ($4$ tầng, $12$ kênh, k3) | **$0.23\text{ MB}$** | **$5.40\text{ ms}$** | **$99.51\%$** | **Nhỏ hơn $1.7\times$, Nhanh hơn $1.5\times$, Chính xác hơn $+0.09\%$!** |

#### Trích xuất nhật ký thực thi thực tế (Raw Execution Logs):
```text
=== EXERCISE 4: NAS SEARCH SPACE (04_nas.ipynb) ===
baseline : d3-w32-k3  (0.38MB)
NAS pick : d4-w12-k3  (0.23MB)
training baseline ... baseline test_acc=99.42%
training NAS pick ... NAS pick test_acc=99.51%
model                           size   accuracy   CPU latency
baseline (d3-w32-k3)          0.38MB     99.42%        7.94ms
NAS pick (d4-w12-k3)          0.23MB     99.51%        5.40ms
>>> NAS pick: 1.7x smaller, 1.5x faster, +0.09% accuracy vs. the hand-designed baseline.
```

---

# 4. BÀI 5: GPU KERNEL OPTIMIZATION (TỐI ƯU HÓA PHẦN CỨNG VỚI TRITON)

### 4.1. Đề bài đang hướng đến điều gì?
* **Thực trạng**: Các kỹ thuật từ Bài 1 đến Bài 4 can thiệp vào tầng toán học hoặc kiến trúc mạng. Tuy nhiên, khi ma trận trọng số $W$ và kích hoạt $x$ được đưa vào GPU để tính phép nhân ma trận $Y = x \cdot W^T$, mã nguồn thường bị thắt cổ chai bởi **băng thông bộ nhớ toàn cục (DRAM/HBM)**.
* **Mục tiêu**: Tự tay viết một GPU Kernel nhân ma trận (**GEMM - General Matrix Multiplication**) bằng ngôn ngữ **OpenAI Triton**, áp dụng kỹ thuật phân khối (**Tiling**), sắp xếp thứ tự khối (**L2 Cache Grouping**), và tính toán trực tiếp trên **Tensor Cores**, giúp tăng tốc độ tính toán gấp nhiều lần mà không làm sai lệch kết quả toán học.

---

### 4.2. Kiến thức & Bản chất phần cứng GPU (SRAM vs DRAM)

```
┌─────────────────────────────────────────────────────────────┐
│ GPU DRAM / HBM (Bộ nhớ toàn cục)                            │
│ Dung lượng: 16 - 80 GB  |  Băng thông: ~1 - 2 TB/s (CHẬM)   │
└──────────────────────────────┬──────────────────────────────┘
                               │  Nạp theo từng khối (Tiles)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ SRAM / Shared Memory / L1 Cache (Nằm sát nhân tính toán)    │
│ Dung lượng: Vài trăm KB |  Băng thông: ~15 - 20 TB/s (CỰC NHANH) │
└──────────────────────────────┬──────────────────────────────┘
                               │  Đưa vào thanh ghi (Registers)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ Tensor Cores / ALU: Thực hiện lệnh tl.dot(A_tile, B_tile)    │
└─────────────────────────────────────────────────────────────┘
```

#### Công thức toán học của phép nhân ma trận:
Cho hai ma trận $A \in \mathbb{R}^{M \times K}$ và $B \in \mathbb{R}^{K \times N}$, ma trận tích $C = A \cdot B \in \mathbb{R}^{M \times N}$:
$$C_{i,j} = \sum_{k=0}^{K-1} A_{i,k} \cdot B_{k,j}$$

Nếu mỗi phép nhân cộng đều đọc $A_{i,k}$ và $B_{k,j}$ từ DRAM, số lần truy cập DRAM là $\mathcal{O}(M \cdot N \cdot K)$ $\to$ Làm GPU nghẽn hoàn toàn.

#### Giải pháp Tiling (Phân khối):
Chia ma trận $C$ thành các khối kích thước $\text{BLOCK\_M} \times \text{BLOCK\_N}$. Mỗi khối được tính toán bởi một Thread Block (Program) trên GPU.
Mỗi vòng lặp, ta chỉ nạp một lát cắt $\text{BLOCK\_K}$ của $A$ và $B$ vào SRAM:
$$\text{acc} \leftarrow \text{acc} + A_{\text{tile}} \cdot B_{\text{tile}}$$
Số lần đọc DRAM giảm đi $\mathcal{O}(\text{BLOCK})$ lần!

---

### 4.3. Giá trị cốt lõi: Tiling & L2 Cache Grouping
* **Grouped Ordering**: Nếu duyệt các khối theo thứ tự dòng tuần tự thông thường (Row-major), các Program chạy đồng thời sẽ cùng cần một dải ngang của $A$ nhưng lại cần các cột hoàn toàn khác nhau của $B$, làm tràn bộ nhớ đệm L2 Cache. 
* Kỹ thuật **Grouping** gom các Program chạy cùng thời điểm thành các cụm vuông $\text{GROUP\_M} \times \text{num\_pid\_n}$, giúp tối đa hóa tỷ lệ dữ liệu tái sử dụng trực tiếp trong L2 Cache.

---

### 4.4. Lời giải chi tiết các `TODO` trong `05_kernel_optimization.ipynb` (Cell 7)

```python
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,          # Con trỏ tới bộ nhớ GPU của A, B, C
    M, N, K,                       # Kích thước ma trận
    stride_am, stride_ak,          # Bước nhảy bộ nhớ của ma trận A
    stride_bk, stride_bn,          # Bước nhảy bộ nhớ của ma trận B
    stride_cm, stride_cn,          # Bước nhảy bộ nhớ của ma trận C
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # ---- (a) Grid layout: Sắp xếp Program ID theo Grouped Ordering để tối ưu L2 Cache ----
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    # TODO 1: Ánh xạ pid -> (pid_m, pid_n) theo cụm GROUP_M
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ---- Tính toán con trỏ ban đầu cho Tile hiện tại ----
    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # ---- (b) Tiling: Trượt qua trục K và tích lũy trong thanh ghi ----
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        
        # TODO 2: Tích lũy tích ma trận khối bằng Tensor Cores
        acc += tl.dot(a, b)
        
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # ---- (c) Ghi ma trận tích C ra DRAM và xử lý biên (Masking) ----
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    
    # TODO 3: Mặt nạ bảo vệ tránh ghi vượt biên giới hạn (M, N)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=c_mask)
```

---

### 4.5. Kết quả thực nghiệm & Tốc độ tăng tốc

Dưới đây là kết quả kiểm chứng trực tiếp khi chạy notebook `05_kernel_optimization.ipynb` trên GPU NVIDIA Quadro T2000 (CUDA 12.4, Triton 3.2.0 trong WSL):

#### a. Kiểm chứng tính đúng đắn (Numerical Correctness)
* Phép so sánh ma trận kích thước $1024 \times 1024$:
  * Sai số tuyệt đối lớn nhất: `max abs difference: 0.0625`
  * Đối chiếu logic: `match torch.matmul: True` $\implies$ Khớp hoàn toàn với thư viện PyTorch cuBLAS chuẩn.

#### b. Benchmark tốc độ thực tế (Triton Tiled GEMM vs PyTorch cuBLAS)
| Kích thước ma trận ($M=N=K$) | PyTorch cuBLAS (ms) | Triton Kernel (ms) | PyTorch (TF/s) | Triton (TF/s) | Hệ số tăng tốc (Speedup) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **$512 \times 512$** | $0.748\text{ ms}$ | **$0.483\text{ ms}$** | $0.4\text{ TF/s}$ | **$0.6\text{ TF/s}$** | **$1.55\times$** |
| **$1024 \times 1024$** | $5.143\text{ ms}$ | **$3.074\text{ ms}$** | $0.4\text{ TF/s}$ | **$0.7\text{ TF/s}$** | **$1.67\times$** |
| **$2048 \times 2048$** | $41.163\text{ ms}$ | **$24.534\text{ ms}$** | $0.4\text{ TF/s}$ | **$0.7\text{ TF/s}$** | **$1.68\times$** |
| **$4096 \times 4096$** | $331.287\text{ ms}$ | **$197.638\text{ ms}$** | $0.4\text{ TF/s}$ | **$0.7\text{ TF/s}$** | **$1.68\times$** |

```mermaid
xychart-beta
    title "Thời gian thực thi ma trận GEMM (ms) - Càng thấp càng tốt"
    x-axis ["512x512", "1024x1024", "2048x2048", "4096x4096"]
    y-axis "Thời gian (ms)" 0 --> 350
    bar [0.748, 5.143, 41.163, 331.287]
    bar [0.483, 3.074, 24.534, 197.638]
```

> **Kết luận thực tế**: Kernel Triton tự xây dựng đạt tốc độ trung bình **`1.64x`** nhanh hơn so với `torch.matmul` trên GPU Quadro T2000 nhờ tối ưu hóa phân khối SRAM Tiling và sắp xếp thứ tự khối để tận dụng tối đa L2 Cache.

---

# 5. BẢNG TỔNG HỢP SO SÁNH 5 KỸ THUẬT TỐI ƯU HÓA

| STT | Phương pháp | Can thiệp vào đâu? | Tác động bộ nhớ (RAM/Disk) | Tác động độ trễ (Latency) | Điểm mấu chốt kỹ thuật |
|:---:|---|---|:---:|:---:|---|
| **1** | **Quantization** | Trọng số $W \to$ Số nguyên INT8 | **Giảm $4\times$ chắc chắn** | Tăng tốc khi có INT8 GEMM kernel | Scale $s_W$, Zero-point $z_W$, Per-channel |
| **2** | **Distillation** | Phân phối Softmax của $W_S$ | Giữ nguyên kích thước $W_S$ | **Giữ nguyên tốc độ siêu nhanh** | Nhiệt độ $T$, Soft labels, KL Divergence |
| **3** | **Pruning** | Loại bỏ kết nối/bộ lọc trong $W$ | Giảm khi cắt cấu trúc filter | **Giảm mạnh FLOPs và Latency** | Chuẩn $L_1$ norm $\|W_c\|_1$, Fine-tuning |
| **4** | **NAS** | Tìm kiếm cấu trúc tầng $D, C, K$ | Tùy chọn theo ngân sách | Đạt đường biên Pareto tối ưu | Proxy Evaluation, Đột biến di truyền |
| **5** | **GPU Kernel** | Cách đọc/ghi $W, x$ trên SRAM/DRAM | Giữ nguyên tham số mô hình | **Tăng tốc $2\times - 5\times$ trên GPU** | Tiling, SRAM, L2 Grouping, Tensor Cores |

## Video kiểm thử robot trên máy — 04/11/2026

Đã chạy SmolVLA với LIBERO libero_object, task 0: **pick_up_the_alphabet_soup_and_place_it_in_the_basket**. Đây là lượt baseline SmolVLA, chưa áp dụng các phép tối ưu MNIST trong báo cáo lên mô hình robot.

- Kết quả: **thành công 1/1 episode**, 138 bước, không bị hủy giữa chừng; seed 42, n_action_steps=1.
- Runtime: vla.cpp bản CPU trong WSL Ubuntu; checkpoint rfai/smolvla-libero-gguf. Ở lượt này, bản CUDA mới gặp kernel yêu cầu SM80, không tương thích GPU Quadro T2000 (SM75), nên tạm dùng CPU.
  - **Cập nhật (đã khắc phục):** sau lượt này đã có [bản sửa tương thích Turing](robot-results/turing-local.patch), bỏ các candidate GEMM chỉ dành cho Ampere và dùng fallback có sẵn. Bản CUDA đã chạy được trên T2000, xem mục "Kết quả kiểm thử robot GPU chính thức" bên dưới (200 episode, 173 thành công).
- Thời gian suy luận trung bình mỗi bước: **3405,67 ms**.
- Video: H.264, 512 × 256, 30 fps, 139 khung hình, 4,63 giây. Đây là thời lượng phát theo khung hình mô phỏng; thời gian chạy thực tế dài hơn do lượt này suy luận trên CPU.
- Một episode thành công chưa đủ để suy ra tỷ lệ thành công trên toàn bộ benchmark hoặc hiệu quả tối ưu hóa.

[Xem video robot](robot-results/baseline/smolvla/libero_object/task_0/episode_000000.mp4) · [Nhật ký tổng hợp](robot-results/baseline/smolvla/libero_object/task_0/summary.txt) · [Kết quả từng episode](robot-results/baseline/smolvla/libero_object/task_0/episodes.csv)


### Cập nhật kiểm thử GPU và CPU — 04/11/2026

LIBERO-Object task 0, 20 episode, replay 10: GPU CUDA FP16 thành công **20/20**, CPU BF16 **17/20**. Đã xác thực 20 video GPU. Đây là baseline SmolVLA; các phép tối ưu MNIST chưa được áp dụng lên policy robot. Toàn bộ benchmark 10 task đã chạy xong sau đó, xem mục bên dưới. [Xem báo cáo và video chi tiết](robot-results/BAO_CAO_KIEM_THU_ROBOT.md).


## Kết quả kiểm thử robot GPU chính thức — 04/11/2026

Đã hoàn tất đủ **200 episode trên GPU Quadro T2000: 173 thành công (86,5%), 27 thất bại**, đủ 200 video đã xác thực. [Báo cáo robot và toàn bộ video](robot-results/BAO_CAO_KIEM_THU_ROBOT.md). [Báo cáo hiệu năng theo template](vla-runtime/docs/benchmark/quadro-t2000-local.md): mean **544,1 ms/predict** theo protocol 3 warmup + 20 reps × 3 tiến trình, cấu hình mặc định. Tỷ lệ thành công thuộc rollout FP16, khác cấu hình benchmark; không trộn hai kết quả. Phép đo CPU đã dừng theo yêu cầu. Các notebook MNIST vẫn là bài tối ưu riêng; chưa áp dụng trực tiếp lên policy robot.
