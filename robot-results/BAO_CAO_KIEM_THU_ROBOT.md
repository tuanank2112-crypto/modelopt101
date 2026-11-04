# Báo cáo kiểm thử robot SmolVLA trên GPU — 04/11/2026

**Kết quả: 173/200 episode thành công (86,5%), 27 thất bại, không episode nào bị hủy giữa bước.** Đã chạy đủ 10 task LIBERO-Object, mỗi task 20 episode; đã xác thực đủ 200 video H.264 bằng ffprobe.

[Xem toàn bộ 200 video](DANH_SACH_VIDEO_GPU.md) · [Bảng kết quả CSV](gpu-summary.csv) · [Báo cáo hiệu năng theo template](../vla-runtime/docs/benchmark/quadro-t2000-local.md)

## Cấu hình kiểm thử

- Thiết bị: NVIDIA Quadro T2000 4 GB, sm_75; WSL Ubuntu 26.04, CUDA toolkit 12.4, driver 580.92.
- Checkpoint: `vrfai/smolvla-libero-gguf`, SHA256 `6fb2d475c98b4c2cef3e27c4eff4e67b483740cbf983fff320a3b8a5e5f74fe8`.
- Runtime: vla.cpp `4a542ed` cộng [bản sửa tương thích Turing](turing-local.patch). Trọng số resident **FP16**; hai camera 512×512; action chunk 50, thực thi 10 hành động rồi dự đoán lại; seed môi trường 42; giới hạn 500 bước/episode.
- Thời gian: xem các mốc TASK_START/TASK_DONE trong [nhật ký đầy đủ](gpu-rollout-complete.log); hoàn tất lúc **15:52:46 ngày 04/11/2026 (UTC+07)**.
- Khởi động server bằng ảnh giả 224 pixel bị từ chối; các episode thực tế dùng ảnh 512 pixel đúng cấu hình. Lỗi warmup này không được tính thành episode hoặc benchmark warmup.

## Kết quả theo task

| Task | Thành công | Tỷ lệ | Client ms/bước môi trường | Dữ liệu |
|---|---:|---:|---:|---|
| 0 — Alphabet soup | 20/20 | 100% | 138.98 | [Kết quả](gpu-f16-libero-object-200/smolvla/libero_object/task_0/episodes.csv) |
| 1 — Cream cheese | 19/20 | 95% | 134.32 | [Kết quả](gpu-f16-libero-object-200/smolvla/libero_object/task_1/episodes.csv) |
| 2 — Salad dressing | 18/20 | 90% | 136.73 | [Kết quả](gpu-f16-libero-object-200/smolvla/libero_object/task_2/episodes.csv) |
| 3 — BBQ sauce | 14/20 | 70% | 128.67 | [Kết quả](gpu-f16-libero-object-200/smolvla/libero_object/task_3/episodes.csv) |
| 4 — Ketchup | 19/20 | 95% | 128.46 | [Kết quả](gpu-f16-libero-object-200/smolvla/libero_object/task_4/episodes.csv) |
| 5 — Tomato sauce | 11/20 | 55% | 140.14 | [Kết quả](gpu-f16-libero-object-200/smolvla/libero_object/task_5/episodes.csv) |
| 6 — Butter | 20/20 | 100% | 136.71 | [Kết quả](gpu-f16-libero-object-200/smolvla/libero_object/task_6/episodes.csv) |
| 7 — Milk | 19/20 | 95% | 139.67 | [Kết quả](gpu-f16-libero-object-200/smolvla/libero_object/task_7/episodes.csv) |
| 8 — Chocolate pudding | 17/20 | 85% | 131.29 | [Kết quả](gpu-f16-libero-object-200/smolvla/libero_object/task_8/episodes.csv) |
| 9 — Orange juice | 16/20 | 80% | 129.87 | [Kết quả](gpu-f16-libero-object-200/smolvla/libero_object/task_9/episodes.csv) |
| **Tổng** | **173/200** | **86,5%** | Không gộp các định nghĩa độ trễ | |

`Client ms/bước môi trường` là số client ghi trong summary.txt khi phát lại 10 hành động, **không phải thời gian một lần dự đoán**. Một episode chứa nhiều lần dự đoán; một task chứa 20 episode. Các lượt hết 500 bước được tính thất bại và video vẫn được giữ.

## Hiệu năng riêng theo template

Đã đo GPU độc lập bằng vla-bench: 3 warmup + 20 lần đo/tiến trình, lặp 3 tiến trình. Chọn tiến trình có mean thấp nhất theo template: **mean 544.1 ms/predict**, min 542.3, p50 543.7, p90 545.4; peak host RSS trên ba tiến trình **450.5 MiB**. Đầu vào tổng hợp gồm 2 ảnh 512×512 và 16 token; cấu hình mặc định, khác FP16 của bộ rollout.

Đã sàng lọc 8 cấu hình GPU. Flash Attention giảm mean dưới 3% trong screen nên không thay mặc định. Số đo này không khẳng định độ chính xác tác vụ của cấu hình mặc định; tỷ lệ 86,5% thuộc **cấu hình rollout FP16**.

WSL không cung cấp VRAM theo từng process (`N/A`); báo cáo không tự điền số ước lượng vào cột này. Mẫu mức dùng toàn GPU trong rollout nằm ở [CSV sử dụng GPU](gpu-utilization-sample.csv), tách khỏi protocol benchmark.

## Phạm vi và giới hạn

- Đây là SmolVLA baseline. Quantization, distillation, pruning, NAS, Triton trong các notebook MNIST chưa được chuyển thành các biến thể policy robot; không gán những phương pháp đó cho các video này.
- Nhiễu suy luận không được ghép cặp cố định; không suy ra chênh lệch chất lượng do backend từ hai lần chạy khác dtype.
- Kết quả CPU cũ của task 0: 17/20 với BF16, chỉ là dữ liệu bổ sung. Bộ CPU bị dừng khi chuyển ưu tiên GPU; **đo hiệu năng CPU tiếp theo cũng đã dừng theo yêu cầu mới nhất**. Các lượt screen CPU dở không được trình bày như benchmark hoàn chỉnh.
- Bản sửa Turing bỏ các candidate chỉ dành cho Ampere để dùng fallback có sẵn. Các archive BitVLA/FoldQuant được biên dịch SM80 để liên kết, không đồng nghĩa các mô hình đó chạy được trên T2000.

## Bằng chứng và tái hiện

- [Log rollout](gpu-rollout-complete.log), [log server](gpu-server-complete.log), [số đo thô benchmark](template-benchmark/results.json).
- Mỗi thư mục task có episodes.csv, summary.txt, video_validation.json và 20 MP4.
- Dùng run_gpu_object_200.sh trong vla-runtime để tái hiện; server CUDA cần `--weight-dtype f16`, client `--n-action-steps 10`. Giữ đủ số episode, task và giới hạn bước như trên.
