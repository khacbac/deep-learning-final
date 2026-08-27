# Báo cáo & nguồn số liệu

**Bản báo cáo nằm ở [`BAO_CAO.md`](BAO_CAO.md)** — 10 mục ánh xạ 1-1 với khung 70/30 của đề bài. Mọi con số
trong đó được trích từ `tables/` bên dưới, không có số nào gõ tay. Dựng bản HTML tự chứa (hình nhúng thành
data URI, không gọi tài nguyên ngoài trừ Google Fonts):

```bash
python report/build_html.py       # -> report/bao_cao.html
python report/check_numbers.py    # đối chiếu 39 con số trong báo cáo với tables/
```

Sửa `BAO_CAO.md` rồi chạy lại hai script trên; **đừng sửa `bao_cao.html` bằng tay** vì nó bị ghi đè.
`check_numbers.py` bắt trường hợp sửa số ở một chỗ mà quên chỗ khác — nó chuẩn hoá dấu phẩy thập phân
kiểu Việt và dấu phân cách nghìn trước khi so, rồi thoát với mã lỗi nếu lệch.

---

Thư mục này là **nguồn duy nhất** để lấy số và hình khi viết báo cáo. Toàn bộ nội dung được
**trích tự động** bằng `extract.py` từ output đã lưu trong
`../notebooks/gastrovision_classification.ipynb`, tức là từ đúng vòng chạy đã sinh ra mọi con số
trong `../RESULTS.md`.

**Vòng chạy (lần thứ 4, 27-08-2026):** profile `gpu-a100` (NVIDIA A100-SXM4-40GB), 30 epoch,
batch 32, `SEEDS = [0,1,2]`, `SPLIT_SEED = 42`, 22 lớp, 7.930 ảnh (test 1.586), quy tắc chốt
`SELECTION_RULE = "top3_tta"`, AMP bfloat16 + TF32.

Điểm macro-F1 **không phụ thuộc phần cứng** — cả 12 lượt chạy chính đều được khôi phục từ `.npz`
rồi tính lại từ logits đã lưu, nên chúng giống hệt vòng T4 trước đó. **Chỉ hai thứ là số của phần
cứng:** bảng độ trễ (`tables/28_*`) và đường val 3 epoch của Gate 0a (`tables/11_*`). Xem
`../RESULTS.md` Gate 0a: cùng seed + cùng code + khác GPU = khác mô hình, nên **σ không bao giờ
được trộn phần cứng**.

---

## ⚠️ KHÔNG lấy số từ `../outputs/` và `../checkpoints/`

Hai thư mục đó (đều bị gitignore) đang chứa kết quả của một **lần chạy smoke trên CPU**:
`../outputs/bang_tong_ket.csv` ghi macro-F1 **0,4129 / 0,3889 / 0,3396 / 0,2870** với `so_seed = 1`
— đó là 12/4/4 ảnh mỗi lớp và 2 epoch, **không phải kết quả của dự án**. Các `.png` / `.npz` / `.pt`
ở đó cũng cùng đợt smoke ấy. Số thật nằm trong thư mục này.

## Hình (từ vòng A100, dữ liệu đầy đủ)

| File | Là gì | Dùng cho phần nào của báo cáo |
| --- | --- | --- |
| `figures/06_eda.png` | Phân bố 22 lớp, trục log — đuôi dài rõ rệt: 1.467 ảnh so với 29 ảnh, mất cân bằng **50,6×** | EDA / mô tả bài toán |
| `figures/18_per_class_va_confusion.png` | Ma trận nhầm lẫn của **P1_coatnet0_288** (mô hình đề xuất), seed 0 | Nhãn & thẩm định, phân tích lỗi |
| `figures/24_duong_hoc_val.png` | Đường học val của cả 4 cấu hình, seed đầu — B0 đỉnh ở epoch 12, S0 ở 7, P1 ở 27 | Lập luận "30 epoch là bảo thủ *với* mô hình đề xuất" |

> Ma trận nhầm lẫn đang tô theo **số đếm thô**, nên đường chéo của các lớp lớn áp hết màu và lỗi ở
> lớp hiếm gần như vô hình. Bản **chuẩn hoá theo hàng** sẽ đọc tốt hơn nhiều cho báo cáo; muốn vẽ
> thì cần các `.npz` per-seed trên Drive, không tái tạo được từ máy local.

**Không có ảnh chụp demo Gradio ở đây** — Gradio hiện ra dưới dạng widget động nên notebook không
lưu ảnh nào. Bằng chứng dạng chữ là `tables/29_demo_gradio.txt` (nạp checkpoint 288 px trên CUDA +
tự kiểm tra trên một ảnh test thật). Nếu báo cáo cần hình thì tự chụp từ ô đang chạy trên Colab.

## Bảng (text output nguyên văn, không chỉnh sửa)

Số thứ tự = **thứ tự cell trong notebook**, nên nó sẽ dịch khi thêm mục mới. Các file đáng dùng
nhất, theo thứ tự lập luận của báo cáo:

| File | Nội dung |
| --- | --- |
| `02_ho_so_chay.txt` | Cấu hình chạy — bằng chứng protocol (epoch / batch / seed / AMP) |
| `04_loc_lop_22.txt`, `05_chia_split.txt` | Luật "> 25 ảnh" → 22 lớp, và chia phân tầng 60:20:20 |
| `06_eda.txt` | Số ảnh từng lớp, tỉ lệ mất cân bằng |
| `07_audit_md5.txt`, `08_audit_gan_trung.txt` | Audit rò rỉ 2 lớp + các cặp gần trùng **khác nhãn** (phát hiện về chất lượng dữ liệu) |
| `11_gate0a_tat_dinh.txt` | Gate 0a — hai lần chạy cùng seed trùng nhau tới 6 chữ số |
| `16_bang_6_quy_tac.txt` | 4 mô hình × 6 quy tắc chọn checkpoint → cơ sở chốt `top3_tta` |
| `17_bootstrap_ci.txt` | Bootstrap CI 95% từng mô hình |
| `18_per_class_va_confusion.txt` | F1 từng lớp của P1 (đối chiếu Table 3 của paper trong `RESULTS.md` §9) |
| `19_donbay_hieuchinh_logit.txt`, `20_donbay_ensemble_kientruc.txt` | Hai đòn bẩy 0-GPU |
| `21_bang_tong_ket.txt` | **Bảng tổng kết chính**: 4 dòng, 3 seed, mean ± σ, CI, so với 0,6504 |
| `22_do_ben_truoc_ro_ri.txt` | Bỏ ảnh test nghi rò rỉ rồi tính lại — Δ ~0,002 |
| `23_he_thong_de_xuat_3seed.txt` | **Tuyên bố "vượt baseline"**: 0,6961 ± 0,0016, CI [0,6548; 0,7245]; và vì sao hiệu chỉnh logit bị loại |
| `26_transfer_learning_log.txt`, `27_transfer_learning.txt` | **Transfer learning — 10% của rubric.** 4 điều kiện freeze/trainable trên cùng DenseNet-121: T1 0,5725 · T2 0,6463 · T3 0,6472 · T4 0,6676 ± 0,0066. File `26` là log 6 quy tắc của từng điều kiện, `27` là bảng so sánh + ngưỡng 2σ + đối chứng Table 2 |
| `28_trien_khai_onnx_do_tre.txt` | Độ trễ / kích thước ONNX **trên A100** (batch 1 so với batch 32) |
| `29_demo_gradio.txt` | Demo §20b nạp được checkpoint thật và tự kiểm tra đúng — phần Deployment |

## Cách tạo lại thư mục này

```bash
python report/extract.py        # chạy từ final-project/
```

Script xoá sạch `tables/` + `figures/` rồi ghi lại từ notebook. Nó nhận diện cell theo **dòng đầu
của source** (notebook được sinh từ `build_notebook.py` nên chuỗi này ổn định) chứ không theo chỉ số
cell, nên thêm cell ở giữa sẽ không làm lệch mapping — và nếu không khớp thì script **báo lỗi** thay
vì ghi ra file sai. Đừng sửa tay các file ở đây: lần chạy `extract.py` tiếp theo sẽ ghi đè hết.

**Sau mỗi phiên Colab:** commit lại `.ipynb`, chạy `python report/extract.py`, rồi cập nhật dòng
"Vòng chạy" ở đầu file này nếu phần cứng thay đổi (số macro-F1 thì không, nhưng bảng độ trễ và
Gate 0a thì có).
