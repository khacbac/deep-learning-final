# Báo cáo & nguồn số liệu

**Bản báo cáo nằm ở [`BAO_CAO.md`](BAO_CAO.md)** — 10 mục ánh xạ 1-1 với khung 70/30 của đề bài. Mọi con số
trong đó được trích từ `tables/` bên dưới, không có số nào gõ tay. Dựng bản HTML tự chứa (hình nhúng thành
data URI, không gọi tài nguyên ngoài trừ Google Fonts):

```bash
python report/build_html.py       # -> report/bao_cao.html
python report/check_numbers.py    # đối chiếu 105 con số trong báo cáo với tables/
```

Sửa `BAO_CAO.md` rồi chạy lại hai script trên; **đừng sửa `bao_cao.html` bằng tay** vì nó bị ghi đè.
`check_numbers.py` bắt trường hợp sửa số ở một chỗ mà quên chỗ khác — nó chuẩn hoá dấu phẩy thập phân
kiểu Việt và dấu phân cách nghìn trước khi so, rồi thoát với mã lỗi nếu lệch.

---

Thư mục này là **nguồn duy nhất** để lấy số và hình khi viết báo cáo. Toàn bộ nội dung được
**trích tự động** bằng `extract.py` từ output đã lưu trong
`../notebooks/final-gastrovision-classification.ipynb`, tức là từ đúng vòng chạy đã sinh ra mọi con số
trong `../RESULTS.md`.

**Vòng chạy (T4 / Kaggle, 31-08-2026, `SESSION = 4`):** profile `gpu-t4` (Tesla T4), `SEEDS = [0,1,2]`,
`SPLIT_SEED = 42`, 22 lớp, 7.930 ảnh (test 1.586), AMP float16 + GradScaler. `B0`/`S0`/`P0`/`P1` 30
epoch; `P2`/`P2b`/`P2c` 80 epoch với công thức hiện đại. Quy tắc chốt **`SELECTION_RULE = "top3"`**.

Phiên 4 **có** resume thật: `tables/12_*` mở đầu bằng `doc lai tu B0_densenet121_seed0.npz (bo qua
huan luyen)`, tức 6 cấu hình của phiên 1 được đọc lại từ `.npz` chứ không train lại — đúng thứ mà ô
checklist cuối file này bắt phải kiểm. Phần **mới** của phiên 4 là `P2c` (1 seed), `P2b` seed 1+2, và
`A1`/`A2` ở bảng `25`. Chi tiết: `../RESULTS.md` §10.10.

> ⚠️ **Đã sửa 31-08-2026 — chỗ này trước đây viết sai.** Câu cũ là: *"Điểm macro-F1 không phụ thuộc
> phần cứng — cả 12 lượt chạy chính đều được khôi phục từ `.npz` rồi tính lại từ logits đã lưu, nên
> chúng giống hệt vòng T4 trước đó."* Mệnh đề sau **đúng**, mệnh đề trước **không suy ra được từ nó**:
> tính lại từ logits đã lưu thì bất biến, nhưng câu đó không kiểm tra rằng resume **đã thực sự xảy
> ra**. Và nó đã không xảy ra — cả 4 cấu hình gốc bị **huấn luyện lại** trên T4, nên mọi con số ở
> `../RESULTS.md` §9 đã bị thay thế và `SELECTION_RULE` tự lật từ `top3_tta` sang `top3`. Chi tiết +
> bằng chứng: `../RESULTS.md` §10.9 phát hiện 5. **Điểm macro-F1 PHỤ THUỘC phần cứng** khi phải huấn
> luyện lại; mức lệch đo được nằm ở `tables-offline/30_lap_lai_a100_vs_t4.txt`.

**σ không bao giờ được trộn phần cứng** (Gate 0a). Trong bộ số hiện tại thì mỗi cấu hình có cả 3
seed trên cùng một loại GPU, nên σ sạch; nhưng các phép so **giữa** vòng 1 (A100) và vòng 2 (T4) thì
không so được trực tiếp — dùng bảng 30 để biết biên độ.

---

## Ba thư mục, đừng lẫn

| Thư mục | Là gì | Dùng khi nào |
| --- | --- | --- |
| `tables/`, `figures/` | Trích tự động từ notebook, **vòng T4 hiện tại** (`SESSION = 4`) | Mặc định — mọi số của báo cáo |
| `tables-a100/`, `figures-a100/` | Bản lưu của vòng **A100 27-08-2026**, trước khi bị train lại | Đúng **2** bảng mà vòng T4 không có: `11` (Gate 0a) và `29` (demo Gradio) — cộng bản A100 của `28` (độ trễ ONNX) và `12`–`15` để đối chiếu ở bảng `30`. **7 file đó là tất cả những gì được dùng; 23 file còn lại chỉ là lưu trữ — đừng trích.** `figures-a100/` còn **2** hình: `06_eda.png` trùng byte với bản `figures/` (EDA không phụ thuộc phần cứng) nên đã xoá 01-09-2026 |
| `tables-offline/` | Tính lại **0 GPU** bằng `offline_tables.py` — bảng `30`–`34` từ `../ckpt-t4/*.npz`, bảng `35`–`36` đọc lại chính `tables/*.txt` | **7 bảng** (`30`–`36`) mà không ô nào của notebook in ra được: xem bảng liệt kê ở cuối file |

`extract.py` **xoá sạch** `tables/` + `figures/` mỗi lần chạy, và chỉ hai thư mục đó — ba thư mục
kia nó không chạm tới.

## ⚠️ KHÔNG lấy số từ `../outputs/` và `../checkpoints/`

Hai thư mục đó **không tồn tại nữa** — đã xoá 01-09-2026. Chúng chứa kết quả của một **lần chạy
smoke trên CPU** (12/4/4 ảnh mỗi lớp, 2 epoch): `../outputs/bang_tong_ket.csv` ghi macro-F1
**0,4129 / 0,3889 / 0,3396 / 0,2870** với `so_seed = 1`, và 352 MB `.onnx` cùng đợt. Suốt một
tháng chúng chỉ làm được đúng một việc là bắt cả hai file README phải cảnh báo về chúng.

Luật vẫn giữ nguyên vì chúng **sẽ mọc lại** ngay lần chạy local kế tiếp: `checkpoints/` và
`outputs/` là những gì lần chạy **gần nhất trên máy này** ghi ra, không phải kết quả dự án.
Số thật nằm trong thư mục này.

Hai thư mục `.npz` thì **giữ**, và chúng khác nhau:

| Thư mục | Là gì | Ai đọc |
| --- | --- | --- |
| `../ckpt-t4/` | logits 3 seed của vòng T4 hiện tại, tải từ Output của phiên Kaggle | `offline_tables.py` (bảng 30–34) |
| `../ckpt-a100/` | logits vòng A100 27-08, giải nén từ `gastrovision-ckpts.zip` cũ. Chứa cả `T1`/`T2`/`T3` mà `ckpt-t4/` **không** có | chưa script nào đọc — giữ để tính lại transfer learning ở 0 GPU nếu cần |

## Hình (vòng T4, dữ liệu đầy đủ)

| File | Là gì | Dùng cho phần nào của báo cáo |
| --- | --- | --- |
| `figures/06_eda.png` | Phân bố 22 lớp, trục log — đuôi dài rõ rệt: 1.467 ảnh so với 29 ảnh, mất cân bằng **50,6×** | EDA / mô tả bài toán |
| `figures/18_per_class_va_confusion.png` | Ma trận nhầm lẫn của **P2_coatnet0_288_modern** (mô hình đề xuất mới), seed 0 — ô tự chọn mô hình tốt nhất nên nó đã đổi theo | Nhãn & thẩm định, phân tích lỗi |
| `figures/24_duong_hoc_val.png` | Đường học val của cả **7** cấu hình, seed đầu — B0 đỉnh ở epoch 6, S0 ở 6, P1 ở 11, **P2 ở 40/80**, còn `P2c` @224 ở **11/80** | Lập luận "30 epoch đủ cho vòng 1, nhưng công thức hiện đại cần 80" — và `P2c` là phản ví dụ đáng nói (mục 3.4) |

> Ma trận nhầm lẫn đang tô theo **số đếm thô**, nên đường chéo của các lớp lớn áp hết màu và lỗi ở
> lớp hiếm gần như vô hình. Bản **chuẩn hoá theo hàng** sẽ đọc tốt hơn nhiều cho báo cáo; muốn vẽ
> thì cần các `.npz` per-seed trên Drive, không tái tạo được từ máy local.

**Không có ảnh chụp demo Gradio ở đây** — Gradio hiện ra dưới dạng widget động nên notebook không
lưu ảnh nào. Bằng chứng dạng chữ là `tables-a100/29_demo_gradio.txt` (nạp checkpoint 288 px trên
CUDA + tự kiểm tra trên một ảnh test thật); **bản T4 của bảng này bị bỏ qua** vì nhánh resume không
copy `.pt`. Nếu báo cáo cần hình thì tự chụp từ ô đang chạy.

## Bảng (text output nguyên văn, không chỉnh sửa)

Số thứ tự = **thứ tự cell trong notebook**, nên nó sẽ dịch khi thêm mục mới. Các file đáng dùng
nhất, theo thứ tự lập luận của báo cáo:

| File | Nội dung |
| --- | --- |
| `02_ho_so_chay.txt` | Cấu hình chạy — bằng chứng protocol (epoch / batch / seed / AMP) |
| `04_loc_lop_22.txt`, `05_chia_split.txt` | Luật "> 25 ảnh" → 22 lớp, và chia phân tầng 60:20:20 |
| `06_eda.txt` | Số ảnh từng lớp, tỉ lệ mất cân bằng |
| `07_audit_md5.txt`, `08_audit_gan_trung.txt` | Audit rò rỉ 2 lớp + các cặp gần trùng **khác nhãn** (phát hiện về chất lượng dữ liệu) |
| `11_gate0a_tat_dinh.txt` | ⚠️ vòng T4 **bỏ qua** Gate 0a (`RUN_DETERMINISM_CHECK` chỉ bật ở `SESSION = "all"`) — lấy bản `tables-a100/11_*`, hai lần chạy cùng seed trùng 6 chữ số |
| `16_bang_6_quy_tac.txt` | 6 mô hình × 6 quy tắc chọn checkpoint → cơ sở chốt **`top3`** (vòng A100 chốt `top3_tta`; xem `RESULTS.md` §10.9 phát hiện 5) |
| `17_bootstrap_ci.txt` | Bootstrap CI 95% từng mô hình |
| `18_per_class_va_confusion.txt` | F1 từng lớp của **P2** — precision 0,810 so với recall 0,690, tức nút thắt còn lại là lớp hiếm (đối chiếu Table 3 của paper trong `RESULTS.md` §9) |
| `19_donbay_hieuchinh_logit.txt`, `20_donbay_ensemble_kientruc.txt` | Hai đòn bẩy 0-GPU |
| `15b_P2_cong_thuc_hien_dai.txt`, `15c_P2_tach_don_bay.txt` | **Bậc P2**: công thức hiện đại (+0,0471 trên CoAtNet @288) và `P2b` tách đòn bẩy công thức khỏi kiến trúc (−0,0024 trên DenseNet @224 — **kết quả âm**) |
| `17b_so_sanh_theo_cap.txt` | **Bậc M**: paired bootstrap + McNemar, in **mọi** seed + dòng `ens3seed` |
| `21_bang_tong_ket.txt` | **Bảng tổng kết chính**: 7 dòng (`B0` `S0` `P0` `P1` `P2` `P2b` × 3 seed + `P2c` × 1), mean ± σ, CI, so với 0,6504 |
| `22_do_ben_truoc_ro_ri.txt` | Bỏ ảnh test nghi rò rỉ rồi tính lại — Δ ~0,002 |
| ~~`23_he_thong_de_xuat_3seed.txt`~~ | ⚠️ **đừng lấy số ở đây.** Nó **đã** in P2 (cảnh báo cũ ở file này nói "vẫn in P1" — sai, sửa 01-09-2026), nhưng ô 19b sinh ra nó **ghim cứng `top3_tta`** trong khi quy tắc chốt là `top3`, nên nó ra **0,7486** còn báo cáo ra **0,7441**. Code đã sửa (`build_notebook.py`, `test_notebook.py` nhóm 8) nhưng notebook chỉ bắt kịp ở phiên Kaggle sau. Số đúng: `tables-offline/31_he_thong_p2_hieu_chinh_logit.txt` |
| `tables-offline/30_*` | Lặp lại A100 ↔ T4 cùng seed: `top3` bền với phần cứng gấp ~4× các quy tắc một-checkpoint |
| `tables-offline/31_*` | **Tuyên bố "vượt baseline"**: P2 + `top3` + hiệu chỉnh logit = **0,7441 ± 0,0088**, CI seed đầu [0,6986; 0,7736] — không chồng lấn 0,6504 |
| `tables-offline/32_*` | F1 từng lớp của P2 đối chiếu **Table 3 của bài báo**, từng lớp một |
| `tables-offline/33_*` | Mô hình đề xuất so với **cả hai** baseline (paired bootstrap + McNemar), không chỉ với `B0` |
| `tables-offline/34_*` | Dư địa còn lại nằm ở lớp nào — cơ sở của mục 4.6 |
| `tables-offline/35_*` | **Bảng 2×2 tương tác**: `công thức × kiến trúc` +0,0468 so với `công thức × độ phân giải` +0,0085 |
| `tables-offline/36_*` | Mất cân bằng (hàm mất mát vs suy luận) và dữ liệu pretrain — đọc `A1`/`A2` cùng với hiệu chỉnh logit |
| `25_ablation_tuy_chon.txt` | **Nhóm C của phiên 4** — `A1` Swin-T pretrain **IN-22k** (0,7028 vs `S0` 0,6774 = **+0,0254**) và `A2` balanced softmax (0,6831 vs `B0` 0,6878 = **−0,0047**, kết quả âm). ⚠️ Bản `tables-a100/25_*` chỉ là một dòng *"bo qua ablation"* — **đừng lấy nó thay bản này** |
| `26_transfer_learning_log.txt`, `27_transfer_learning.txt` | **Transfer learning — 10% của rubric.** 4 điều kiện freeze/trainable trên cùng DenseNet-121, vòng T4: T1 0,5674 · T2 0,6596 · T3 0,6394 · T4 0,6780 ± 0,0073 (bản A100: 0,5725 / 0,6463 / 0,6472 / 0,6676). File `26` là log 6 quy tắc của từng điều kiện, `27` là bảng so sánh + ngưỡng 2σ + đối chứng Table 2 |
| `28_trien_khai_onnx_do_tre.txt` | Độ trễ / kích thước ONNX **trên T4** (batch 1 so với batch 32) — bản A100 ở `tables-a100/28_*`, **không trộn hai bảng** |
| `29_demo_gradio.txt` | ⚠️ vòng T4 **bỏ qua** (không có `.pt` sau khi resume) — lấy bản `tables-a100/29_*`; cách sửa ở `RESULTS.md` §10.9 |

## Cách tạo lại thư mục này

```bash
python report/extract.py           # tables/ + figures/  <- output trong .ipynb
python report/offline_tables.py    # tables-offline/     <- ../ckpt-t4/*.npz, 0 GPU
```

`extract.py` xoá sạch `tables/` + `figures/` rồi ghi lại từ notebook. Nó nhận diện cell theo **dòng
đầu của source** (notebook được sinh từ `build_notebook.py` nên chuỗi này ổn định) chứ không theo chỉ
số cell, nên thêm cell ở giữa sẽ không làm lệch mapping — và nếu không khớp thì script **báo lỗi**
thay vì ghi ra file sai. Đừng sửa tay các file ở đây: lần chạy tiếp theo sẽ ghi đè hết.

> ⚠️ Nó `sys.exit` **sau khi** đã xoá `tables/`, nên một prefix lệch = mất sạch bảng cũ (khôi phục
> bằng `git checkout report/tables`). Sửa 31-08-2026: 4 prefix đã lệch vì notebook đổi
> `RUN_DETERMINISM_CHECK = True` / `RUN_P1_288 = True` sang `SESSION_FLAGS.get(...)`.

**Sau mỗi phiên Kaggle/Colab:**

1. commit lại `.ipynb`;
2. **kiểm tra `tables/12_*` xem có dòng `doc lai tu ... (bo qua huan luyen)` không** — nếu không có
   thì phiên đó đã **train lại** cấu hình gốc chứ không resume, và mọi con số cũ bị thay thế. Đây
   đúng là thứ đã đi lọt ở vòng T4 (`RESULTS.md` §10.9 phát hiện 5);
3. `python report/extract.py` rồi `python report/offline_tables.py`;
4. `python report/check_numbers.py` để bắt chỗ báo cáo còn trích số cũ;
5. cập nhật dòng "Vòng chạy" ở đầu file này. **Số macro-F1 CÓ đổi khi đổi phần cứng nếu phải huấn
   luyện lại** — chỉ bất biến khi thực sự đọc lại từ `.npz`.
