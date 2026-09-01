# Vượt baseline công bố trên GastroVision

**Phân loại 22 lớp ảnh nội soi tiêu hoá — CNN vs Transformer vs Hybrid, và một bài học về cách đo**

Môn AIN501 · Artificial Intelligence (Deep Learning) · MSE FSB
Nhóm 2 thành viên · Thành viên A (track CNN): Hồ Khắc Bác · Thành viên B (track Transformer): Lê Trọng Quang
Ngày 31-08-2026

---

## Tóm tắt

Bài báo GastroVision (arXiv 2307.08140, Table 2) công bố **macro-F1 = 0,6504** cho DenseNet-121
pretrained trên 22 lớp / 7.930 ảnh. Chúng tôi làm hai việc và **tách rời chúng**:

**Kết quả 1 — tái lập được baseline, nhưng lỏng hơn ta muốn.** Dưới đúng giao thức của bài báo (một
checkpoint tốt nhất theo val, không TTA), DenseNet-121 của chúng tôi cho **0,6686 ± 0,0234** trên
3 seed. Chênh **+0,0182**, tức **0,78 σ** — nằm trong một độ lệch chuẩn, nên tái lập được; nhưng
±0,023 là một dải rộng, và mục 1.4 nói rõ vì sao con số này *không* nên được đọc là "trùng khít".
Chúng tôi đạt nó trong **30 epoch** thay vì 150 epoch của họ.

**Kết quả 2 — vượt baseline.** Hệ thống đề xuất là **CoAtNet-0 @288 + công thức huấn luyện hiện đại
+ ensemble top-3 checkpoint + hiệu chỉnh logit**, cho **macro-F1 = 0,7441 ± 0,0088** trên 3 seed,
**+0,0937** so với 0,6504, với bootstrap CI 95% = **[0,6986; 0,7736]** — khoảng tin cậy **không
chứa** 0,6504. Micro-F1 (= accuracy) đạt **0,850** so với 0,8203 của bài báo.

**Luận điểm chính của báo cáo, và nó không phải luận điểm chúng tôi dự định tìm.** Khi phân rã toàn
bộ mức tăng thành các thành phần đã đo riêng, dưới **một** quy tắc chấm điểm duy nhất:

| Lever | Δ macro-F1 | Vượt ngưỡng phân giải ±0,035? |
|---|---|---|
| **Công thức huấn luyện hiện đại** (`P1` → `P2`) | **+0,0443** | ✅ **có** |
| Hiệu chỉnh logit lúc suy luận (0 epoch) | +0,0143 | ❌ không |
| Quy tắc chọn checkpoint `best` → `top3` (0 epoch) | +0,0094 | ❌ không |
| Độ phân giải 224 → 288 *(dưới công thức cũ; dưới công thức mới là **−0,0006**)* | +0,0041 | ❌ không |
| **Đổi kiến trúc** DenseNet-121 → CoAtNet-0 *(đứng một mình)* | **+0,0034** | ❌ không |
| | **tổng +0,0755** | |

**Hai lever mà chúng tôi đặt ra để kiểm định — kiến trúc và độ phân giải — mua được +0,003 và +0,004,
không phân biệt được với số không.** Thứ thực sự trả tiền là **công thức huấn luyện** (cosine + LLRD
+ EMA + mixup + 80 epoch, +0,0443), và sau đó là **cách đo** (hiệu chỉnh logit + quy tắc checkpoint,
cộng lại +0,0237 với **0 epoch phụ**). Đó là kết luận của báo cáo này, và nó trái với kỳ vọng ban đầu
"nút thắt là backbone".

Nhưng ngay cả kết luận đó cũng có một điều kiện: **công thức hiện đại KHÔNG phổ quát.** Cùng công
thức ấy áp lên backbone của bài báo (DenseNet-121 @224) cho **−0,0110** trên 3 seed, cả 3/3 đều âm —
nó làm DenseNet *tệ đi*. Phiên `SESSION = 4` chạy nốt ô thứ tư của bảng 2×2 (`P2c` = CoAtNet-0 @224
+ cùng công thức) và tách được vế đúng:

| Số hạng, quy tắc `top3` | Δ | Vượt ±0,035? |
|---|---|---|
| **công thức × kiến trúc** (cùng 224 px) | **+0,0468** | ✅ **có** |
| công thức × độ phân giải (cùng backbone hybrid) | +0,0085 | ❌ không |

Trên **cùng seed 0**, mỗi yếu tố đứng một mình đều ≤ 0 (kiến trúc −0,0160; công thức −0,0043) còn
ghép lại thì +0,0294 — tức **đây là một tương tác, không phải tổng của hai lever**. Điều này cũng lật
một phỏng đoán ngầm: **288 px không phải thứ làm nên chuyện.** Dưới công thức mới, 288 so với 224 cho
**−0,0006** trong khi tốn **1,70×** chi phí tính toán, nên cấu hình *nên triển khai* là `P2c` @224
(mục 3.4) — dù con số *được báo cáo* vẫn là `P2`, vốn có đủ 3 seed.

**Một phát hiện về phương pháp mà chúng tôi không tìm mà gặp** (mục 2.5): 12 lượt chạy của vòng 1 tồn
tại trên **hai loại GPU** vào hai thời điểm khác nhau, cùng code / cùng split / cùng seed. So hai bộ
số đó cho một phép lặp lại 4 cấu hình × 3 seed × 6 quy tắc, và nó cho thấy **xếp hạng kiến trúc của
vòng 1 không sống nổi qua một lần đổi phần cứng**, trong khi quy tắc `top3` thì bền gấp ~4 lần các
quy tắc một-checkpoint. Đây là bằng chứng độc lập, đến từ một hướng hoàn toàn khác, cho chính điều mà
mục 5.3 phải phát biểu: *ở 3 seed, không được quyền nói Swin-T thắng DenseNet-121.*

Hai con số phụ, mỗi con số là **một dòng riêng** vì chúng tiêu nhiều lần huấn luyện: ensemble 3 seed
của mô hình đề xuất = **0,7587** (CI [0,7110; 0,7924]), còn ensemble nhiều kiến trúc chọn trên val
= **0,7130** — **thấp hơn** mô hình đơn tốt nhất, tức là một **kết quả âm** (mục 5.5).

Toàn bộ số trong báo cáo này được trích tự động từ output đã lưu của notebook (`report/extract.py`)
hoặc tính lại từ logits đã lưu (`report/offline_tables.py`), không có con số nào gõ tay.

---

### Bản đồ: mấy mô hình, mấy baseline, và mọi thứ còn lại là gì

**Có đúng 2 baseline và 1 mô hình đề xuất.** Mọi tên phương pháp khác xuất hiện trong báo cáo đều
thuộc một trong ba loại phía dưới vạch, và **không loại nào là baseline**:

| | Là gì | Cụ thể | Nhóm có huấn luyện? |
|---|---|---|---|
| **Baseline 1 — CNN tham chiếu** | mô hình của bài báo; phải khớp 0,6504 | `B0` DenseNet-121 @224, 30 epoch | ✅ 3 seed |
| **Baseline 2 — Transformer** | nhánh bài báo **không có**; đóng góp của nhóm | `S0` Swin-T @224, 30 epoch | ✅ 3 seed |
| **Mô hình đề xuất** | hybrid + độ phân giải + công thức + cách đo | `P2` CoAtNet-0 @288 + công thức hiện đại, 80 epoch, + ensemble top-3 checkpoint + hiệu chỉnh logit | ✅ 3 seed |
| — | — | — | — |
| *Table 2 của bài báo* | **số công bố**, trích để biết phải vượt con số nào | ResNet-50 · ResNet-152 · EfficientNet-B0 · DenseNet-169 · ResNet-50 pretrained · DenseNet-121 | ❌ số của họ, không phải của nhóm |
| *Bước trung gian* | tồn tại để cô lập **đúng một** biến | `P0` CoAtNet-0 @224 (cô lập kiến trúc) · `P1` CoAtNet-0 @288 (cô lập độ phân giải) | ✅ 3 seed mỗi cái |
| *Đối chứng công thức* | cô lập "công thức" khỏi "kiến trúc" | `P2b` DenseNet-121 @224 + **cùng** công thức hiện đại | ✅ 3 seed |
| *Ô thứ tư của bảng 2×2* | cô lập "công thức × kiến trúc" khỏi "công thức × độ phân giải" | `P2c` CoAtNet-0 @224 + **cùng** công thức hiện đại | ✅ 1 seed |
| *Lever / ablation* | thay đổi trên một mô hình **đã có** | **nhóm B (3 seed):** 6 quy tắc chọn checkpoint · hiệu chỉnh logit · lọc rò rỉ · ensemble — **nhóm C (1 seed, T4):** `A1` pretrain IN-22k · `A2` balanced softmax — **nhóm A (1 seed, giao thức cũ):** augmentation mạnh · Balanced-Softmax · cRT | ✅ nhưng không phải baseline |
| *4 điều kiện transfer learning* | **cùng** DenseNet-121, chỉ khác độ sâu được phép học | T1 probe · T2 nửa dưới · T3 progressive · T4 = chính `B0` | ✅ 1 seed mỗi điều kiện (T4: 3) |

Cộng lại: **3 kiến trúc** (DenseNet-121, Swin-T, CoAtNet-0), **7 cấu hình** = 19 lượt huấn luyện
(6 cấu hình × 3 seed + `P2c` × 1), cộng 3 lượt cho transfer learning, 2 lượt ablation `A1`/`A2` và
4 lượt lever ở giao thức cũ.

**Vì sao báo cáo dài như vậy:** khung điểm của đề bài (mục 6, trang 15) đặt **70% trọng số vào phần
dữ liệu**, và riêng mục "Xử lý dữ liệu" 30% yêu cầu *"mỗi bước phải có ablation chứng minh nó có
ích, kể cả kết quả âm"*, còn mục transfer learning 10% yêu cầu *"bảng so sánh"* bốn điều kiện. Phần
lớn độ dài nằm ở các bảng ablation bắt buộc đó, chứ không ở phần mô hình — vốn chỉ có ba kiến trúc.

---

## 1 · Bài toán & baseline công bố — **5%**

### 1.1 Dữ liệu và bài toán

GastroVision là bộ ảnh nội soi tiêu hoá đa lớp, thu tại Bærum Hospital (Na Uy) và Karolinska
University Hospital (Thuỵ Điển). Bản phát hành công khai có **8.000 ảnh / 27 lớp**; theo đúng luật
của bài báo (giữ lớp có **> 25 ảnh**) còn **7.930 ảnh / 22 lớp**. Bài toán là **phân loại đơn nhãn
22 lớp**, và metric chính là **macro-F1** vì mất cân bằng lớp lên tới 50,6×.

Các lớp trải từ **giải phẫu bình thường** (Normal stomach, Pylorus, Cecum, Ileocecal valve) qua
**bệnh lý** (Colon polyps, Colorectal cancer, Esophagitis, Barrett's esophagus) tới **dấu vết can
thiệp** (Dyed-lifted-polyps, Dyed-resection-margins, Resected polyps, Accessory tools). Đây là lý do
macro-F1 là chỉ số đúng: một hệ thống sàng lọc bỏ sót lớp bệnh lý hiếm thì vô dụng, dù accuracy tổng
vẫn đẹp nhờ các lớp giải phẫu bình thường đông đảo.

### 1.2 Bảng baseline gốc — **đã tự kiểm chứng tại nguồn**

Chúng tôi không trích lại con số từ blog hay từ bảng thứ cấp mà **mở đúng arXiv 2307.08140 và đọc
Table 2 (trang 11)**:

| Mô hình (Table 2) | macro-F1 công bố | Chế độ fine-tune |
|---|---|---|
| ResNet-152 | 0,4496 | chỉ lớp cuối |
| EfficientNet-B0 | 0,4519 | chỉ lớp cuối |
| DenseNet-169 | 0,4883 | chỉ lớp cuối |
| ResNet-50 | 0,6176 | toàn mạng |
| **DenseNet-121** | **0,6504** | **toàn mạng** |

**Baseline mạnh nhất là DenseNet-121 = 0,6504**, và đó là con số phải vượt. Nguyên tắc 5 của đề bài
đòi tái lập baseline mạnh nhất *trước* khi tuyên bố vượt, nên toàn bộ mục 1.4 dành cho việc đó.

Một chi tiết của Table 2 được tái dùng làm **đối chứng miễn phí cho mục 6**: bảng đó trộn hai chế độ
fine-tune, và khoảng cách giữa chúng là **~0,16 macro-F1** trên đúng split này.

### 1.3 Giao thức — của họ và của chúng tôi

| Hạng mục | Bài báo (§4.1) | Của chúng tôi | |
|---|---|---|---|
| Split | phân tầng **60:20:20** | phân tầng 60:20:20, `SPLIT_SEED = 42` | ✅ |
| Số lớp | **22** (chỉ lớp có > 25 ảnh) | 22, có `assert NUM_CLASSES == 22` | ✅ |
| Số ảnh | 7.930 / 8.000 | 7.930 | ✅ |
| Cỡ tập test | **1.586** | 1.586 | ✅ |
| Đầu vào | 224 × 224 | 224 (`B0`/`S0`/`P0`/`P2b`/`P2c`), 288 (`P1`/`P2`) | ✅ |
| Augmentation | random rotation + random hflip | **chỉ hflip** cho vòng 1; `P2`/`P2b`/`P2c` thêm mixup + RandAugment nhẹ (mục 3.2) | ⚠️ khác |
| Optimiser | Adam, lr 1e-4, **ReduceLROnPlateau** | vòng 1: AdamW 1e-4, LR không đổi · `P2`: AdamW + **cosine + warmup 5 + LLRD 0,75 + EMA** | ⚠️ khác |
| **Số epoch** | **150** | **30** (vòng 1) · **80** (`P2`/`P2b`/`P2c`) | ⚠️ khác |
| **Số lần chạy** | **1 lần, không seed, không sai số** | 3 seed, mean ± σ + bootstrap CI | ⚠️ khác |
| Phần cứng | NVIDIA TITAN Xp | **Tesla T4** (Kaggle) — xem mục 2.5 | — |

**Về nguyên tắc "dùng đúng split gốc":** bài báo mô tả split 60:20:20 phân tầng nhưng **không phát
hành file split**. Chúng tôi tự chia, công bố seed (`SPLIT_SEED = 42`) và **kiểm chứng lại split bằng
Table 3 của họ** — bảng đó liệt kê support per-class trên tập test, tổng đúng bằng 1.586. Đối chiếu
với phân bố test của chúng tôi: **16 trong 22 lớp khớp chính xác**, 6 lớp lệch ±1 ảnh (Colon polyps
163/164, Colorectal cancer 29/28, Esophagitis 22/21, Normal mucosa LB 293/294, Pylorus 78/79,
Retroflex rectum 14/13) — đúng kiểu sai số làm tròn của bộ chia phân tầng, và tổng thì trùng khít.
Chúng tôi đánh giá trên một tập test **cùng thành phần**, không chỉ cùng kích cỡ.

### 1.4 Tái lập baseline — con số trung thực

| | macro-F1 | Ghi chú |
|---|---|---|
| Bài báo, DenseNet-121, 150 epoch | 0,6504 | 1 lần chạy, không sai số |
| **Của chúng tôi, DenseNet-121, 30 epoch, 3 seed, cùng quy tắc chọn checkpoint (`best`)** | **0,6686 ± 0,0234** | chênh **+0,0182 = 0,78 σ** |

Đây là **con số tái lập**, và chúng tôi cố tình *không* dùng 0,6780 (giá trị của cùng mô hình dưới
quy tắc `top3`) để gọi là "tái lập bài báo" — vì bài báo không dùng ensemble checkpoint. Trộn hai
quy tắc là cách dễ nhất để tự lừa mình, và mục 3.5 sẽ cho thấy phần chênh đó lớn cỡ nào.

**Ba điều phải nói thẳng về dòng này, vì nó là dòng chống đỡ cho toàn bộ báo cáo.**

**Thứ nhất, chúng tôi *vượt* baseline ở đây chứ không *trùng* nó**, và +0,018 là một chênh lệch cần
giải thích chứ không phải một chiến thắng. Giải thích khả dĩ nhất nằm ngay ở cột σ: **±0,0234 là độ
lệch lớn nhất trong mọi cấu hình của báo cáo**, và nó bị chi phối bởi một seed duy nhất (seed 0 cho
0,7008, so với 0,6461 và 0,6589 ở hai seed còn lại). Quy tắc `best` chọn **một** checkpoint theo đỉnh
val, nên nó thừa hưởng toàn bộ độ nhiễu của val — đó chính là chẩn đoán mà mục 3.5 xây quanh.

**Thứ hai, một bản trước của báo cáo này ghi con số tái lập là 0,6491 ± 0,0124 (chênh −0,0013)**, và
đó là số thật, đo trên A100 hồi 26-08. Bộ trọng số đó không còn tồn tại (mục 2.5), nên chúng tôi
**không** trích nó nữa. Nhưng việc *hai* lần huấn luyện độc lập cùng code / cùng split / cùng seed
cho 0,6491 và 0,6686 lại nói một điều hữu ích: **0,6504 nằm giữa hai lần đo của chúng tôi.** Đó là
bằng chứng tái lập mạnh hơn bất kỳ điểm ước lượng đơn lẻ nào.

**Thứ ba, 0,6504 là một lần chạy không có thanh sai số.** Chính DenseNet-121 của chúng tôi, dưới quy
tắc của họ, tán ±0,0234 qua 3 seed. Vậy cách đọc trung thực là 0,6504 mang theo một khoảng bất định
**không được công bố**, ít nhất cỡ ±0,02. Hệ quả cho báo cáo: câu **"chúng tôi vượt 0,6504" phải được
chống đỡ bằng một khoảng tin cậy không chứa nó**, chứ không phải bằng phép trừ hai điểm ước lượng —
và mục 5.3 cho thấy chỉ **một** trong sáu cấu hình đạt tiêu chuẩn đó.

Việc tái lập được ở 30 epoch cũng **giải quyết luôn phản biện hiển nhiên về ngân sách epoch**:
DenseNet-121 đạt đỉnh val ở **epoch 6/30** (mục 5.4), nên 120 epoch còn lại chưa bao giờ có cơ hội
giúp nó. Giao thức 30-epoch là **công bằng với baseline**. Nó chỉ **bảo thủ với công thức hiện đại**,
và đó chính là lý do `P2` được cho 80 epoch — một chênh lệch được nêu rõ ở mục 3.2 chứ không làm mờ.

---

## 2 · Phân tích dữ liệu (EDA) — **20%**

### 2.1 Từ 8.000 ảnh xuống 7.930, từ 27 lớp xuống 22

```
Quet duoc 8006 anh / 29 lop
Loc lop > 25 anh  ->  22 lop / 7930 anh   (bo 7 "lop", 76 anh)
Toan bo du lieu: train=4758  val=1586  test=1586
```

Luật lọc là **của bài báo**, không phải của chúng tôi: giữ lớp có > 25 ảnh. Năm lớp thật bị bỏ có
1–25 ảnh mỗi lớp. Việc quét thư mục được làm **đệ quy** vì bản phát hành lồng thư mục theo vị trí
giải phẫu; một phép quét một tầng sẽ tìm thấy 0 ảnh và đó là cái bẫy đầu tiên của bộ dữ liệu này.

> ⚠️ **8.006 chứ không phải 8.000, và 29 "lớp" chứ không phải 27 — cái bẫy thứ hai của phép quét đệ
> quy.** Ở phiên `SESSION = 4`, thư mục **Output của chính phiên trước** được gắn vào làm input, nên
> phép quét đệ quy nhặt luôn `outputs/` và `__results___files/` (3 ảnh mỗi thư mục — là các **hình vẽ
> matplotlib** mà notebook đã xuất ra) và tính chúng thành hai lớp. Luật > 25 ảnh của bài báo loại
> cả hai, nên **split, danh sách 22 lớp, 7.930 ảnh và mọi bảng dữ liệu đều không đổi** — đối chiếu
> `tables/05_chia_split.txt`, `06_eda.txt`, `09_bo_danh_gia.txt`: byte-identical với vòng trước. Nếu
> một trong hai thư mục đó có > 25 file thì split đã âm thầm đổi. Đây là lý do luật lọc phải chạy
> **trước** khi chia, và là một rủi ro có thật của cơ chế resume qua thư mục input.

### 2.2 Mất cân bằng 50,6× và cái giá của nó

| | ảnh | train | val | test |
|---|---|---|---|---|
| Normal mucosa and vascular pattern in the large bowel | 1.467 | 880 | 293 | 294 |
| Accessory tools | 1.266 | 760 | 253 | 253 |
| … | | | | |
| Mucosal inflammation large bowel | 29 | 17 | 6 | **6** |
| Colon diverticula | 29 | 17 | 6 | **6** |

**Tỉ lệ mất cân bằng: 50,6×. Hai trong 22 lớp có dưới 10 ảnh test.**

Đây là con số quan trọng nhất của toàn bộ phần EDA, vì nó quyết định **nền nhiễu của mọi phép đo phía
sau**. macro-F1 lấy trung bình 22 lớp với trọng số bằng nhau, nên một lớp 6 ảnh test có trọng số
1/22 = **0,0455** trong con số cuối cùng. Đoán đúng thêm **một** ảnh trong lớp đó làm F1 của lớp nhảy
~0,15, kéo macro-F1 đi ~0,007. Đoán đúng thêm hai ảnh: ~0,02.

**Hệ quả trực tiếp:** nền nhiễu của một phép đo đơn trên bộ test này là cỡ **±0,02–0,05**, tức **lớn
hơn phần lớn hiệu ứng mà chúng tôi muốn đo**. Toàn bộ mục 3 và mục 5 được thiết kế quanh sự thật đó,
và mục 4.5 định lượng nó chính xác.

![Phân bố 22 lớp](figures/06_eda.png)

### 2.3 Audit rò rỉ dữ liệu — hai lớp kiểm tra

Với ảnh nội soi, nhiều frame của cùng một ca là chuyện thường, nên rò rỉ giữa các tập là rủi ro
nghiêm trọng nhất về tính toàn vẹn. Hai lớp kiểm tra, cả hai chạy trên **toàn bộ 7.930 ảnh**:

| Lớp | Phương pháp | Kết quả |
|---|---|---|
| **1. Trùng byte** | MD5 toàn bộ 7.930 ảnh | **1 nhóm trùng vắt qua split**: `Colon polyps`, một ảnh xuất hiện ở **train và val**. Không nhóm nào chạm tập test |
| **2. Gần trùng** | cosine ≥ 0,98 trên embedding MobileNetV3-Small @160px | **9 cặp vắt qua split**, ảnh hưởng **6/1.586 = 0,38%** tập test |

**9 cặp trên 7.930 ảnh là ~0,1%** — quá ít để dịch chuyển macro-F1. Nhưng mục 3.6 **chứng minh** điều
đó bằng cách tính lại sau khi bỏ các ảnh test bị ảnh hưởng, thay vì chỉ khẳng định.

### 2.4 Phát hiện mạnh hơn cả rò rỉ: **nhãn nghi ngờ**

Cùng phép quét cosine trả về một thứ giá trị hơn: các cặp ảnh **gần như trùng khít nhưng mang hai
nhãn khác nhau**.

| Cặp | cosine | Vấn đề |
|---|---|---|
| `val/Esophagitis/N2DaTmFs.jpg` ≡ `test/Normal esophagus/WdSYgDiw.jpg` | **1,0000** | cùng một ảnh, hai nhãn **loại trừ nhau về mặt lâm sàng** |
| `train/Accessory tools/…` ≡ `val/Gastric polyps/…` | 0,9998 | cùng một ảnh, hai nhãn khác nhau |

**Đây là kết quả về chất lượng dữ liệu, không phải về rò rỉ**, và nó đặt một **trần cứng** lên
macro-F1 khả đạt của *bất kỳ* mô hình nào trên bộ này: không mô hình nào có thể đúng đồng thời trên
hai ảnh giống nhau mang hai nhãn khác nhau. Cặp đầu đặc biệt đáng lưu ý vì Esophagitis (viêm) và
Normal esophagus (bình thường) là ranh giới chẩn đoán quan trọng nhất trong nhóm thực quản — và
Esophagitis đúng là một trong 8 lớp yếu nhất của mô hình chúng tôi (mục 4.3).

### 2.5 Tính toàn vẹn phép đo: tất định **theo từng GPU** — và một phép lặp lại ngoài kế hoạch

Trước khi tin bất kỳ con số nào, chúng tôi chạy cùng một seed **hai lần** và so đường val 3 epoch
(Gate 0a). Kết quả qua ba lần đo (`tables-a100/11_gate0a_tat_dinh.txt`):

| Lần đo | Phần cứng | Đường val macro-F1 (3 epoch) |
|---|---|---|
| 26-08, lần 1 & 2 | A100 | `[0.428608, 0.549646, 0.551532]` — **trùng khít 6 chữ số** |
| 27-08 | T4 | `[0.430615, 0.540379, 0.541930]` — **khác** |
| 27-08 (vòng cuối) | A100 | `[0.428608, 0.549646, 0.551532]` — **trùng khít lại** |

**Ba lần đo, một quy tắc: đường val là hàm của GPU, và lặp lại được trên đúng GPU đó.** Tất định giữ
được **trong** một thiết bị, không giữ được **giữa** các thiết bị: cùng seed + cùng code + khác GPU =
một mô hình khác. Hệ quả thực tế: **σ không bao giờ được trộn phần cứng.** Trong báo cáo này, mỗi cấu
hình có cả 3 seed trên **cùng một loại** GPU, nên mọi σ đều sạch.

#### Điều Gate 0a không đo được, và một tai nạn đã đo hộ

Gate 0a đo lệch phần cứng trên **3 epoch**, và ra ~0,010 — nhỏ, dễ khiến người ta yên tâm. Nhưng câu
hỏi thật là: sau **30 epoch** cộng thêm một bước **chọn checkpoint theo val**, lệch đó tích luỹ bao
nhiêu? Gate 0a không trả lời được.

Vòng 2 của dự án chuyển từ Colab (hết compute unit) sang Kaggle, và trong quá trình đó **bốn cấu hình
của vòng 1 đã bị huấn luyện lại trên T4** thay vì được nạp lại từ `.npz` như dự định. Đó là một sai
sót về quy trình, đã ghi lại đầy đủ ở `../RESULTS.md` §10.9. Nhưng nó để lại một thứ không ai bỏ tiền
ra mua được: **một phép lặp lại hoàn chỉnh** — cùng code, cùng split, cùng seed, 4 cấu hình × 3 seed
× 6 quy tắc, trên hai loại GPU (`tables-offline/30_lap_lai_a100_vs_t4.txt`):

| Quy tắc chấm điểm | \|chênh trung bình\| trên 4 cấu hình | \|lệch\| lớn nhất ở một seed |
|---|---|---|
| **`top3`** | **0,0046** | 0,0263 |
| `top3_tta` | 0,0092 | 0,0310 |
| `smooth` | 0,0153 | 0,0710 |
| `best` | 0,0182 | 0,0435 |
| `smooth_tta` | 0,0185 | 0,0717 |
| `best_tta` | 0,0185 | 0,0428 |

**Ba điều đọc ra được, và cả ba đều vào báo cáo:**

1. **Lệch phần cứng lớn hơn Gate 0a ước rất nhiều.** 0,010 ở 3 epoch trở thành **0,04–0,07** ở 30
   epoch + chọn checkpoint. Cơ chế rõ ràng: hai quỹ đạo huấn luyện phân kỳ dần, và quy tắc `best`
   chọn *một điểm* trên quỹ đạo đó nên nó khuếch đại phân kỳ.
2. **`top3` bền với phần cứng gấp ~4 lần** các quy tắc một-checkpoint. Đây là **lý lẽ thứ ba, độc
   lập** cho quy tắc đó, bên cạnh "0 epoch" và "giảm σ" ở mục 3.5.
3. **Xếp hạng kiến trúc của vòng 1 không sống nổi qua một lần đổi máy.** Dưới `top3_tta`, A100 cho
   `P1 > S0 > P0 > B0`; T4 lật thành `B0 > S0 > P1 > P0`. Dưới `top3` thì thứ tự **giữ được** (`P1`
   nhất, `B0` bét ở cả hai máy). Và biên độ 0,015–0,019 của các quy tắc một-checkpoint **lớn hơn cả
   đòn bẩy kiến trúc** mà báo cáo đo được (+0,0034 ở mục 5.3).

Điểm 3 là điều đáng nói nhất, vì nó là **một kiểm chứng độc lập cho tính khiêm tốn của mục 5.3**: ở
đó chúng tôi buộc phải viết *"không tách được Swin-T khỏi DenseNet-121"* chỉ vì CI chồng lấn. Giờ có
bằng chứng trực tiếp — đổi phần cứng là đủ để đảo thứ tự xếp hạng. **Một xếp hạng bị đảo bởi việc đổi
GPU thì không phải một xếp hạng.**

Ngược lại, nó cũng cho biết kết luận nào **an toàn**: +0,0443 của công thức hiện đại (mục 3.2) lớn
hơn mọi hiệu ứng phần cứng trong bảng trên, và `B0`/`P0`/`P1`/`P2`/`P2b`/`P2c` đều trên T4 — nên phép so đó
không vắt qua hai loại máy.

---

## 3 · Xử lý dữ liệu — **30%**

Đây là hạng mục nặng điểm nhất, và **kết luận của nó phần lớn là âm**. Đề bài yêu cầu "mỗi bước phải
có ablation chứng minh nó có ích, kể cả kết quả âm"; chúng tôi đo và phát hiện **phần lớn các bước
được sách vở khuyến nghị đều không giúp gì** trên bộ dữ liệu này, còn thứ giúp được nhiều nhất lại
chỉ giúp cho **một** kiến trúc. Theo nguyên tắc 5 của đề bài, chúng tôi báo cáo đúng như thế.

### 3.0 Cảnh báo về nguồn số — đọc trước bảng tổng kết

Các ablation trong mục này đến từ **ba giao thức khác nhau**, và trộn chúng là sai:

| Nhóm | Giao thức | Trích dẫn được không? |
|---|---|---|
| **A.** Augmentation mạnh, Balanced-Softmax, cRT, 288px-trên-DenseNet | **1 seed**, quy tắc `best`, **trước khi sửa tất định**, trọng số không còn tồn tại | ⚠️ Chỉ trích **kết luận** ("đều phẳng"), **không trích con số** như số đo chính xác |
| **B.** Quy tắc checkpoint + TTA, độ phân giải, lọc rò rỉ, hiệu chỉnh logit, transfer learning | **3 seed** (lọc rò rỉ và transfer: 1 seed), quy tắc `top3`, cùng một vòng chạy T4 | ✅ Đây là số của báo cáo |
| **C.** Công thức huấn luyện hiện đại (`P2`), đối chứng `P2b`, và ô 2×2 `P2c` | **3 seed** cho `P2` và `P2b` / **1 seed** cho `P2c`, 80 epoch, quy tắc `top3`, cùng vòng T4 với nhóm B | ✅ Số của báo cáo; `P2c` chỉ 1 seed nên phép so nào dùng nó đều được đọc **trên cùng seed 0** |

Lý do nhóm A vẫn được giữ lại: mọi Δ của nó (−0,035 / −0,007 / −0,013 / −0,017) đều **nhỏ hơn nền
nhiễu run-to-run** (~±0,02–0,05, mục 2.2), nên kết luận *"không lever nào trong nhóm này thắng được
baseline trơn"* đứng vững **bất kể** con số cụ thể. Đó là một phát biểu yếu hơn, và đúng hơn.

### 3.1 Chuẩn hoá

ImageNet mean/std (`[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]`) cho **cả ba kiến trúc**. Đây
không phải lựa chọn tuỳ ý: cả ba đều khởi tạo từ trọng số pretrained ImageNet, và các thống kê chuẩn
hoá là **một phần của hợp đồng với bộ trọng số đó**. Dùng mean/std tính trên GastroVision sẽ đặt đầu
vào lệch khỏi phân bố mà các bộ lọc đã học, và mục 6 cho thấy chính các lớp đầu là phần nhạy cảm
nhất. Không có ablation ở đây vì lựa chọn bị ràng buộc, không phải vì chúng tôi bỏ qua.

### 3.2 Augmentation & công thức huấn luyện — lever **duy nhất** vượt ngưỡng phân giải

Ba công thức được cài đặt:

| | Nội dung |
|---|---|
| **`plain`** (vòng 1, mọi số của `B0`/`S0`/`P0`/`P1`) | `Resize(sz, sz)` → `RandomHorizontalFlip` → `Normalize` |
| **`strong`** (nhóm A, đã đo, đã loại) | thêm `RandomRotation(10)`, `ColorJitter(0.15, 0.15, 0.15, 0)`, `RandomErasing(p=0.1)` |
| **`modern`** (vòng 2, `P2`/`P2b`/`P2c`) | `plain` + RandAugment nhẹ + **mixup**, cùng với cosine LR + warmup 5 epoch + **LLRD 0,75** + **EMA** (cửa sổ ~5 epoch), 80 epoch |

**Luận cứ theo domain cho từng phép của `strong`:**

* **Lật ngang — giữ.** Ống tiêu hoá không có tính đối xứng gương cố định so với khung hình: hướng
  camera phụ thuộc thao tác của người nội soi, nên ảnh lật ngang vẫn là một ảnh nội soi hợp lệ.
* **Xoay ±10° — loại.** Cùng lý do trên thì xoay nhẹ cũng hợp lệ về mặt vật lý, nhưng `Resize` cố
  định khung vuông khiến phép xoay đưa **viền đen nội suy** vào mép ảnh, và mép đen là một artifact
  đặc trưng của ảnh nội soi — tức là ta đang bơm thêm một tín hiệu giả tương quan với augmentation.
* **ColorJitter — loại, và đây là phép nguy hiểm nhất.** Màu niêm mạc **là dấu hiệu chẩn đoán**:
  Esophagitis, Barrett's esophagus, Blood in lumen đều được nhận ra phần lớn qua sắc đỏ/hồng và ranh
  giới màu. Nhiễu sáng/độ tương phản/độ bão hoà ±15% chồng lên chính chiều thông tin dùng để phân
  biệt bệnh lý với niêm mạc bình thường. Với ảnh tự nhiên đây là augmentation vô hại; với ảnh nội soi
  nó xoá đặc trưng.
* **RandomErasing — loại.** Nhiều lớp được xác định bởi **một cấu trúc cục bộ nhỏ** (một polyp, một
  dụng cụ). Che một ô vuông ngẫu nhiên có xác suất không nhỏ xoá đúng vùng chứa nhãn, tạo cặp
  (ảnh không còn bằng chứng → nhãn cũ) = nhiễu nhãn nhân tạo.

#### Số đo — bốn ô của một bảng 2×2

Phiên `SESSION = 4` chạy nốt ô còn trống (`P2c` = CoAtNet-0 @224 + công thức hiện đại) và bù `P2b`
lên đủ 3 seed. Nguồn: `tables-offline/35_bang_2x2_tuong_tac.txt`, quy tắc `top3`.

| So sánh | Đổi đúng thứ gì | Seed | Δ macro-F1 |
|---|---|---|---|
| **`P1` → `P2`** (CoAtNet-0 @288) | `plain`/30 epoch → `modern`/80 epoch | 3 vs 3 | **+0,0443** |
| **`P0` → `P2c`** (CoAtNet-0 @224) | `plain`/30 epoch → `modern`/80 epoch | 3 vs 1 | **+0,0358** |
| **`B0` → `P2b`** (DenseNet-121 @224) | `plain`/30 epoch → `modern`/80 epoch | 3 vs 3 | **−0,0110** |
| `B0` → `B2` (nhóm A) | `plain` → `strong` + recipe 2-giai-đoạn | 1 | −0,035 |

**Dòng đầu là hiệu ứng lớn nhất của toàn bộ dự án**, và là **lever duy nhất** vượt ngưỡng phân giải
±0,035 của phép đo. Nó được xác nhận theo cặp trên từng seed (`tables/17b_so_sanh_theo_cap.txt`): CI
của Δ không chứa 0 ở seed 1 và seed 2 và ở dòng ensemble-3-seed, McNemar p = 0,0203 / 0,0005.

Giờ mỗi phép so cố định đúng một biến, nên **hai phép trừ tách được hai thành phần**:

* **công thức × kiến trúc** (cùng 224 px): +0,0358 − (−0,0110) = **+0,0468**
* **công thức × độ phân giải** (cùng backbone hybrid): +0,0443 − (+0,0358) = **+0,0085**

Chỉ số hạng thứ nhất vượt ngưỡng phân giải ±0,035. Số hạng thứ hai nhỏ hơn ngưỡng **4 lần**.

#### Đây là một **tương tác**, không phải tổng của hai lever độc lập

So trên cùng seed 0 — phép so chặt hơn, vì `P2c` mới có 1 seed:

| @224, seed 0 | công thức cũ | công thức mới | Δ |
|---|---|---|---|
| DenseNet-121 (CNN) | 0,6878 | 0,6835 | **−0,0043** |
| CoAtNet-0 (hybrid) | 0,6718 | **0,7172** | **+0,0454** |

* đổi **một mình** kiến trúc (giữ công thức cũ): 0,6718 − 0,6878 = **−0,0160**
* đổi **một mình** công thức (giữ CNN): 0,6835 − 0,6878 = **−0,0043**
* đổi **cả hai**: 0,7172 − 0,6878 = **+0,0294**

Từng yếu tố một mình đều **≤ 0**; ghép lại mới dương. Số hạng tương tác = 0,0454 − (−0,0043) =
**+0,0497**. Đây là dạng kết quả mà một bảng 2×2 sinh ra được còn hai phép so lẻ thì không — và là
lý do phải chạy `P2c` chứ không dừng lại ở `P2b`.

**Và nó không phụ thuộc quy tắc chấm điểm** — điều đáng kiểm, vì mục 2.5 vừa cho thấy quy tắc chấm
điểm có thể đảo cả một xếp hạng:

| Cách đọc cùng một đòn bẩy | Quy tắc | Δ |
|---|---|---|
| trung bình các Δ ghép cặp **từng seed** (`tables/17b_*`) | `top3` | **+0,0443** |
| hiệu của hai **trung bình 3 seed** (`tables-offline/35_*` mục D) | `top3_tta` | **+0,0471** |

Hai con số là hai đại lượng khác nhau, không phải hai lần đo cùng một thứ — nhưng cả hai đều vượt
0,035, nên kết luận đứng vững dưới cả hai cách đọc. *(Bản đầu của báo cáo này gán nhãn +0,0443 cho
`top3_tta`; `report/check_numbers.py` bắt được sai lệch đó.)*

Quan trọng hơn: **cả hai số hạng phân rã cũng không đổi kết luận khi đổi quy tắc.** Đọc lại toàn bộ
bảng 2×2 dưới `top3_tta` (`tables-offline/35_*` mục D) cho `công thức × kiến trúc` = **+0,0529** (so
với +0,0468) và `công thức × độ phân giải` = **+0,0083** (so với +0,0085) — vế thứ nhất vẫn vượt
±0,035, vế thứ hai vẫn không. Phép kiểm này không thừa: mục 2.5 cho thấy đổi quy tắc chấm điểm từng
đủ để **đảo cả một xếp hạng kiến trúc**.

**`P2b` là lý do không được phát biểu dòng đầu như một quy luật chung, và giờ nó xếp hạng được.**
Đủ 3 seed, cả **3/3 đều âm** (−0,0043 / −0,0086 / −0,0200 — `tables/17b_*`), σ = 0,0119, và thấp hơn
cả trên val: `P2b` đạt đỉnh 0,6539 so với 0,6600 của `B0`. |Δ| = 0,0110 vẫn dưới ±0,035 nên **không**
được nói "công thức làm hại DenseNet-121"; được nói **"chắc chắn không giúp"** — và thế là đủ để bác
bỏ *"công thức hiện đại là lever phổ quát"*. Ở bản trước `P2b` mới có 1 seed và chỉ đọc được **dấu**.

**Lỗ trong lập luận ở bản trước đã được lấp.** Khi `P1`/`P2` so ở 288 px còn `B0`/`P2b` so ở 224 px
thì hai phép so không chia sẻ độ phân giải, nên phát biểu mạnh nhất được phép khi đó chỉ là *công
thức × (kiến trúc **hoặc** độ phân giải)*. `P2c` đặt cả hai phép so về cùng 224 px và trả lời dứt
khoát: vế đúng là **kiến trúc** (+0,0468), không phải độ phân giải (+0,0085) — **ngược với** phỏng
đoán ngầm trước đó rằng 288 px là thứ làm nên chuyện. Xem thêm mục 3.4.

**Vì sao đây vẫn là một kết quả tốt cho hạng mục 30%:** đề bài đòi ablation kèm kết quả âm. Bảng 2×2
cho **hai** kết quả âm sạch — `P2b` (công thức trên CNN, 3 seed, −0,0110) và ô 288-dưới-công-thức-mới
ở mục 3.4 (−0,0006) — mỗi cái đổi đúng một biến, cùng phần cứng, cùng split, cùng seed, cùng số
epoch. Chúng phủ định đúng hai giả thuyết hấp dẫn nhất mà `P2` gợi ra.

Chênh so với giao thức bài báo: họ dùng rotation + hflip, vòng 1 của chúng tôi **chỉ hflip**. Chúng
tôi ghi rõ chênh này ở bảng mục 1.3 thay vì làm mờ nó.

### 3.3 Cân bằng lớp — ba phương pháp lúc huấn luyện, cả ba phẳng

Với mất cân bằng 50,6× thì đây là chỗ sách vở hứa hẹn nhiều nhất. Chúng tôi thử ở **hai tầng khác
nhau** của bài toán:

| Phương pháp | Tác động ở tầng nào | Δ so với baseline (1 seed) |
|---|---|---|
| **Balanced-Softmax** (nhóm A) | hàm mất mát — re-weight theo tần suất lớp | **−0,007** (phẳng) |
| **Balanced-Softmax** (`A2`, đo lại trên T4, `top3`) | như trên, cùng split/seed/epoch với `B0` | **−0,0047** (phẳng) |
| **cRT (decoupled classifier retraining)** | biên quyết định — đóng băng đặc trưng, huấn luyện lại classifier trên sampling cân bằng | **−0,013** (phẳng) |
| Công thức nâng cao + aug mạnh | tổng hợp | −0,035 (hồi quy, mục 3.2) |

Dòng thứ hai là phép đo lại của dòng thứ nhất trên phần cứng và giao thức hiện hành
(`tables-offline/36_mat_can_bang_va_pretrain.txt`): **−0,007 và −0,0047 trên hai lần chạy độc lập**,
cùng dấu, cùng độ lớn. Đây là một trong số ít lever của dự án **lặp lại được** — chỉ tiếc là lặp lại
một kết quả âm.

**Kết luận, và nó là một kết quả chứ không phải một thất bại:** trên GastroVision, **can thiệp cân
bằng lớp ở tầng mất mát và tầng classifier lúc *huấn luyện* đều không dịch chuyển macro-F1**. Nút
thắt không nằm ở chỗ mô hình "không được thưởng đủ" cho lớp hiếm — nó nằm ở **biểu diễn đặc trưng**:
17 ảnh train không đủ để học một đặc trưng tốt, và không có cách đánh lại trọng số nào tạo ra thông
tin chưa từng có.

**Nhưng có một ngoại lệ quan trọng, và nó ở tầng thứ ba: lúc *suy luận*.** Hiệu chỉnh logit (mục 3.7)
cũng nhắm vào đúng thiên lệch mất cân bằng ấy, nhưng nó sửa **sau khi** mô hình đã học xong — và trên
`P2` nó mua được **+0,0143** với τ\* ổn định trên cả 3 seed và σ co lại.

Phép so sạch nhất là đặt hai thứ đó lên **cùng một backbone, cùng split, cùng seed 0, cùng 30 epoch,
cùng quy tắc `top3`** — điều mà `A2` cho phép làm lần đầu:

| Chữa mất cân bằng ở đâu | DenseNet-121, seed 0 | Δ |
|---|---|---|
| không chữa (`B0`, cross-entropy) | 0,6878 | — |
| **trong hàm mất mát** (`A2`, balanced softmax) | 0,6831 | **−0,0047** |
| **lúc suy luận** (`B0` + hiệu chỉnh logit, τ\* = 0,4) | **0,7091** | **+0,0213** |

Cùng một mục tiêu, hai chỗ sửa, hai dấu ngược nhau — và chỗ sửa thắng lại là chỗ tốn **0 epoch**.
Trên `P2` cùng lever ấy cho +0,0276 với τ\* = 0,5. Đối chiếu đó cho một phát biểu sắc hơn phát biểu
ban đầu của chúng tôi:

> Trên bộ dữ liệu này, can thiệp mất cân bằng **lúc huấn luyện** không giúp gì, còn can thiệp **lúc
> suy luận** thì giúp. Nghĩa là thiên lệch mất cân bằng có thật và có thể sửa, nhưng nó **không phải
> một vấn đề tối ưu hoá** — nó là một phép hiệu chỉnh prior ở đầu ra. Với ~17 ảnh train, việc bắt
> mô hình học lại biên quyết định là đòi hỏi thứ dữ liệu không có; việc dịch biên đó bằng một tham số
> đo trên val thì rẻ và hiệu quả.

Suy luận này **được kiểm chứng độc lập ở mục 4.4** từ chiều thứ ba: một backbone mạnh hơn với công
thức tốt hơn *có* dịch chuyển đúng các lớp hiếm — 90,4% mức cải thiện so với bài báo đến từ 15 lớp
hiếm. Ba lối tiếp cận hoàn toàn khác nhau chỉ về cùng một chẩn đoán.

### 3.4 Độ phân giải 224 → 288 — không phân giải được, và dưới công thức mới thì **bằng 0**

| So sánh | Điều kiện | Δ macro-F1 | σ |
|---|---|---|---|
| `P0` → `P1` (224 → 288, **công thức cũ**) | **3 seed**, `top3` | **+0,0041** | 0,0100 → 0,0068 |
| `P2c` → `P2` (224 → 288, **công thức mới**) | seed 0, `top3` | **−0,0006** | — |
| `B0` → `B5` (DenseNet-121: 224 → 288) | 1 seed, `best` (nhóm A) | −0,017 | — |

**+0,0041 nhỏ hơn ngưỡng phân giải ±0,035 gần 9 lần, và nhỏ hơn cả σ của chính nó** — nên ngay dưới
công thức cũ, lever này đã không phân giải được. Dưới công thức mới nó **biến mất hẳn**: trên cùng
seed 0, `P2c` @224 đạt 0,7172 còn `P2` @288 đạt 0,7166, tức **−0,0006**
(`tables-offline/35_bang_2x2_tuong_tac.txt`, mục C).

Và giá thì đo được: mục 7.1 cho **8,74 ms/ảnh** ở 288 px so với **5,13 ms** ở 224 px khi chạy batch
32 — **1,70× chi phí tính toán để đổi lấy −0,0006 macro-F1**.

> **Vì vậy cấu hình *nên triển khai* là `P2c` (CoAtNet-0 @224 + công thức hiện đại), không phải
> `P2`.** Ngoài chuyện rẻ hơn 1,70×, nó chạy ở **đúng độ phân giải của baseline**, nên phép so với
> DenseNet-121 @224 sạch cả biến độ phân giải — chỉ còn kiến trúc và công thức. CI của nó cũng không
> chồng lấn 0,6504: **[0,6664; 0,7568]**, McNemar p = 0,0018 so với `B0`.
>
> Con số **được báo cáo** vẫn là `P2`, vì `P2` có đủ 3 seed và σ đã biết, còn `P2c` mới có 1 seed.
> Hai vai trò khác nhau: `P2` là kết quả *đã đo đủ*, `P2c` là cấu hình *nên triển khai*. Hợp nhất hai
> vai trò đó chỉ cần 2 seed nữa của `P2c` — khoảng 3,3 giờ T4.

> **Một bản trước của báo cáo này ghi lever độ phân giải là +0,0143 với σ = 0,0016**, và gọi đó là
> "lever xử lý dữ liệu duy nhất có hiệu quả". Con số đó đo trên A100 và không còn nguồn (mục 2.5).
> Việc cùng một lever cho +0,0143 trên một phần cứng và +0,0041 trên phần cứng khác **chính là** hiệu
> ứng mà bảng ở mục 2.5 định lượng — và nó là bài học đắt nhất của dự án: **một hiệu ứng nhỏ hơn biên
> độ lặp lại của chính phép đo thì không phải một hiệu ứng.**

Bài học lặp lại lần thứ hai ở đây: cùng một lever, đo dưới hai công thức huấn luyện khác nhau, cho
+0,0041 và −0,0006. Một hiệu ứng nhỏ hơn ngưỡng phân giải thì **đổi dấu tuỳ ngữ cảnh** — nên nó
không được cộng vào bất kỳ đường tích luỹ nào (mục 5.3).

### 3.5 Quy tắc chọn checkpoint — 0 epoch, phổ quát, và bền với phần cứng

Xuất phát từ chẩn đoán ở mục 2.2: nếu val nhỏ và nhiễu, thì **chọn một checkpoint theo đỉnh val là
một phép đo nhiễu**, không phải một phép chọn mô hình. Chúng tôi cài `Tracker` giữ lại nhiều trạng
thái trong *một* lần huấn luyện, để **một lần train sinh ra 6 con số test**:

* 3 cách chọn: `best` (đỉnh val thô) · `smooth` (đỉnh của val làm trơn 3 epoch) · `top3` (ensemble
  logits của 3 checkpoint val cao nhất)
* × 2: có / không **TTA lật ngang**

Kết quả trên **6 mô hình × 3 seed** (bảng đầy đủ, `report/tables/16_bang_6_quy_tac.txt`):

| Mô hình | `best` | `smooth` | **`top3`** | `best_tta` | `smooth_tta` | `top3_tta` |
|---|---|---|---|---|---|---|
| B0_densenet121 | 0,6686 | 0,6685 | **0,6780** | 0,6671 | 0,6770 | 0,6863 |
| S0_swin_t | 0,6599 | 0,6662 | **0,6813** | 0,6619 | 0,6602 | 0,6843 |
| P0_coatnet0 | 0,6800 | 0,6597 | **0,6814** | 0,6799 | 0,6610 | 0,6766 |
| P1_coatnet0_288 | 0,6424 | 0,6541 | **0,6855** | 0,6464 | 0,6580 | 0,6841 |
| *hạng trung bình (1 = tốt nhất)* | 4,50 | 4,50 | **1,50** | 4,50 | 4,00 | 2,00 |
| — | — | — | — | — | — | — |
| P2_coatnet0_288_modern *(chỉ được chấm)* | 0,7222 | 0,7270 | **0,7298** | 0,7199 | 0,7284 | 0,7312 |
| P2b_densenet121_modern *(chỉ được chấm)* | 0,6611 | 0,6584 | **0,6670** | 0,6676 | 0,6727 | 0,6722 |
| P2c_coatnet0_224_modern *(1 seed, chỉ được chấm)* | 0,7071 | 0,7130 | **0,7172** | 0,7154 | 0,7074 | 0,7154 |

**Chỉ 4 cấu hình gốc được bỏ phiếu chọn quy tắc**, và đây là một quyết định thiết kế được khoá cứng
trong code: nếu để `P2`/`P2b`/`P2c` tham gia bỏ phiếu thì việc thêm một bậc mới có thể **lật quy tắc** và
làm lệch mọi con số đã báo cáo ở các mục trước mà không ai thấy. Các bậc mới được **chấm** dưới quy
tắc đã chốt, không được **chọn** nó.

`top3` thắng hạng (1,50 so với 2,00 của `top3_tta`) → **`SELECTION_RULE = "top3"` cho toàn bộ báo
cáo**. Điểm quan trọng về phương pháp: chúng tôi chốt **một** quy tắc rồi áp cho **mọi** dòng, thay
vì chọn quy tắc tốt nhất cho từng mô hình — cách sau là rò rỉ quy trình.

Hiệu ứng `best` → `top3`, dưới cùng một quy tắc, **trên cả bốn kiến trúc bỏ phiếu**:

| Mô hình | Δ (`best` → `top3`) | σ dưới `best` | σ dưới `top3` |
|---|---|---|---|
| B0_densenet121 | **+0,0094** | 0,0234 | **0,0073** |
| S0_swin_t | **+0,0214** | 0,0115 | **0,0081** |
| P0_coatnet0 | **+0,0014** | 0,0123 | 0,0100 |
| P1_coatnet0_288 | **+0,0431** | 0,0180 | **0,0068** |
| **trung bình** | **+0,0188** | **0,0163** | **0,0080** |

**Ba tính chất, và không tính chất nào là "hiệu ứng lớn":**

1. **Phổ quát.** Dương trên **cả bốn** kiến trúc, cùng một chiều — dấu hiệu của một hiệu ứng thật
   chứ không phải một lần bốc thăm may. (Thêm `P2` thì là 5/5.)
2. **Miễn phí lúc huấn luyện.** 0 epoch phụ. Đây là lever có giá trị trên mỗi phút GPU cao nhất của
   dự án.
3. **Giảm phương sai gần một nửa** (σ trung bình 0,0163 → 0,0080), và **bền với phần cứng gấp ~4
   lần** các quy tắc một-checkpoint (mục 2.5).

Nhưng phải nói thẳng: **+0,0188 vẫn dưới ngưỡng phân giải ±0,035.** Nên cách phát biểu đúng không
phải *"quy tắc checkpoint mua nhiều macro-F1"* mà là: **nó là cách rẻ nhất để làm cho các phép đo
khác trở nên đo được.** Giá trị của nó nằm ở cột σ, không ở cột Δ. Toàn bộ mục 3.4 là ví dụ: lever độ
phân giải chỉ có thể được thảo luận sau khi σ đã bị nén xuống 0,007.

### 3.6 Lọc rò rỉ — ablation khẳng định kết quả không đến từ rò rỉ

Bỏ 6/1.586 ảnh test có bản sao byte-identical hoặc cosine ≥ 0,98 trong train/val, rồi tính lại.
**Bảng này chạy trên seed 0** — ô §19b đọc lại logits của seed đầu — nên cột "đầy đủ" là điểm của
riêng seed 0, khác với mean 3 seed ở mục 3.5 và 5.3. Thứ cần đọc ở đây là **cột Δ**:

| Mô hình | đầy đủ | đã lọc | Δ |
|---|---|---|---|
| B0_densenet121 | 0,6878 | 0,6855 | −0,0023 |
| S0_swin_t | 0,6774 | 0,6753 | −0,0021 |
| P0_coatnet0 | 0,6718 | 0,6701 | −0,0017 |
| P1_coatnet0_288 | 0,6837 | 0,6812 | −0,0025 |
| **P2_coatnet0_288_modern** | **0,7166** | **0,7149** | **−0,0016** |
| P2b_densenet121_modern | 0,6835 | 0,6818 | −0,0016 |
| P2c_coatnet0_224_modern | 0,7172 | 0,7155 | −0,0018 |

Mọi Δ đều ≈ −0,002, tức **nhỏ hơn σ giữa các seed** của mọi cấu hình (0,0068–0,0100 dưới `top3`) từ
3 đến 6 lần, và nhỏ hơn ngưỡng phân giải ±0,035 khoảng 20 lần. Riêng với mô hình đề xuất, Δ = −0,0016
so với mức tăng +0,0937 là tỉ lệ **1:59**. Một dịch chuyển 0,002 gây ra bởi 6 ảnh không phải thứ tạo
ra +0,0937. **Rò rỉ không phải thứ sinh ra kết quả** — và giờ đây đó là một phát biểu đã đo, không
phải một lời trấn an.

### 3.7 Hiệu chỉnh logit — lever mất-cân-bằng **duy nhất** có hiệu quả, và một cái bẫy về σ

Hiệu chỉnh logit (logit adjustment, Menon et al. 2021) trừ `τ · log(prior lớp)` khỏi logits lúc suy
luận, với τ tinh chỉnh **trên val**. Nó là lever 0-GPU hứa hẹn nhất cho dữ liệu đuôi dài — và câu
chuyện của nó trong dự án này là bài học phương pháp sắc nhất mà chúng tôi thu được.

Đo trên `P1` và `P2`, **cùng quy tắc `top3`**, cả 3 seed
(`tables-offline/31_he_thong_p2_hieu_chinh_logit.txt`):

| Mô hình | τ\* mỗi seed | raw | sau hiệu chỉnh | Δ | σ |
|---|---|---|---|---|---|
| `P1_coatnet0_288` | 0,2 / **0,0** / 0,1 | 0,6855 | 0,6907 | +0,0052 | 0,0068 → 0,0095 (**×1,40**) |
| **`P2_coatnet0_288_modern`** | **0,5 / 0,5 / 0,3** | 0,7298 | **0,7441** | **+0,0143** | 0,0096 → **0,0088** (**co lại**) |

Notebook thực thi một **tiêu chí đặt trước** cho việc giữ hay bỏ thành phần này: chỉ giữ nếu nó vừa
nâng trung bình **vừa** không làm σ phồng quá 1,5×. **Cả hai đều đạt** — nhưng chúng đạt theo hai
cách rất khác nhau, và sự khác biệt đó là thứ đáng báo cáo:

* Trên `P1`, mức tăng **+0,0052** nhỏ hơn σ của chính nó, τ\* dao động 0,0–0,2 và **một seed chọn
  τ\* = 0, tức chọn không hiệu chỉnh gì**. Nó đạt tiêu chí ở sát mép (×1,40 so với ngưỡng ×1,50).
* Trên `P2`, mức tăng **+0,0143** gấp 1,5 lần σ, τ\* ổn định và xa 0 (0,3–0,5), và σ **co lại** thay
  vì phồng. Đây là dạng bằng chứng khác về chất, không chỉ khác về độ lớn.

Chi tiết từng seed của `P2`, vì đây là thành phần duy nhất trong hệ thống có siêu tham số được đo lại
theo từng seed và vì vậy phải tự chứng minh:

| seed | `top3` | τ\* | sau hiệu chỉnh | Δ |
|---|---|---|---|---|
| 0 | 0,7166 | 0,5 | **0,7442** | **+0,0276** |
| 1 | 0,7335 | 0,5 | 0,7332 | −0,0003 |
| 2 | 0,7392 | 0,3 | 0,7548 | **+0,0156** |
| **mean ± σ** | **0,7298 ± 0,0096** | — | **0,7441 ± 0,0088** | **+0,0143** |

Seed 1 gần như không được lợi, nên +0,0143 **không** đồng đều trên ba seed — nhưng khác với vòng A100
ở dưới, không seed nào bị *hại*, và τ\* không seed nào rơi về 0.

Nên phát biểu đúng là: **hiệu chỉnh logit hoạt động trên cả hai, nhưng chỉ trên `P2` nó là một hiệu
ứng chứ không phải một dao động.** Chúng tôi giữ nó trong hệ thống đề xuất, và ghi rõ rằng +0,0143
vẫn **dưới** ngưỡng phân giải ±0,035 — nó nằm trong nhóm "cách đo", không phải nhóm "hiệu ứng lớn".

**Nó cũng là lever hiếm hoi *không* kén cấu hình.** Trên cả 7 cấu hình, đo ở seed 0 dưới `top3`
(`tables/19_donbay_hieuchinh_logit.txt`): `B0` +0,0213 · `P0` +0,0117 · `P1` +0,0176 · `P2` +0,0276 ·
`P2b` +0,0065 · `P2c` +0,0145 — **6/7 dương, 0/7 âm**, và trường hợp còn lại (`S0`) là τ\* = 0, tức
thủ tục tự chọn "không hiệu chỉnh gì" chứ không phải chọn sai. Đối lập hẳn với công thức hiện đại,
vốn đổi dấu khi đổi kiến trúc (mục 3.2).

#### Cái bẫy về σ — và vì sao nó là bài học chứ không phải một chú thích

Ở **vòng A100**, cùng lever này bị **loại** khỏi hệ thống. Số của vòng đó (`../RESULTS.md` §10.8):
τ\* = 0,9 / 0,5 / 0,0, mức tăng +0,0011 / +0,0271 / +0,0000, trung bình +0,0094 nhưng **σ phồng từ
0,0016 lên 0,0139 — gấp 8,7 lần**, nên tiêu chí ×1,5 chặn nó lại. Kết luận ghi vào nhật ký lúc đó là
*"không có gì để ăn ở đây"*, và nó được dùng làm lý do hạ kỳ vọng của cả một nhánh kế hoạch.

**Kết luận đó không sai về số liệu, nhưng nó sai về đối tượng.** Mẫu số của phép kiểm là σ, và ở vòng
A100 σ của `P1` là **0,0016** — nhỏ bất thường, nhỏ hơn 4–7 lần cùng cấu hình đo trên T4 (0,0068).
Với một mẫu số như thế, **bất kỳ** lever nào thêm dù chỉ chút phương sai cũng thất bại phép kiểm
×1,5. Tiêu chí không phát hiện ra rằng lever này vô dụng; nó phát hiện ra rằng σ của lần chạy đó nhỏ
một cách không tái lập được — và mục 2.5 cho biết chính xác vì sao: **σ = 0,0016 là một đặc tính của
lần chạy đó trên phần cứng đó, không phải của cấu hình.**

Đây là **lần thứ ba** cùng một bài học xuất hiện trong báo cáo này, mỗi lần từ một hướng khác:

| Lần | Ở đâu | Đại lượng bị đọc quá tin |
|---|---|---|
| 1 | mục 2.5 | **xếp hạng** bốn kiến trúc — đảo khi đổi GPU |
| 2 | mục 3.4 | **lever độ phân giải** — +0,0143 trên một phần cứng, +0,0041 trên phần cứng khác |
| 3 | mục 3.7 | **σ**, tức chính cái thước dùng để chấp nhận hay loại các lever khác |

Lần thứ ba là lần đáng lo nhất, vì nó không làm sai một con số — nó làm sai **một quyết định**. Chúng
tôi đã suýt cắt hẳn một nhánh kế hoạch dựa trên một phép kiểm mà mẫu số của nó không tái lập được.
Biện pháp đã áp dụng: **đo lại lever trên mô hình mới thay vì thừa hưởng kết luận cũ**, và đó đúng là
việc đã cứu chúng tôi ở đây.

#### Vì sao `P2` hưởng lợi nhiều hơn — một giả thuyết, không phải một kết luận

Bảng per-class của `P2` (mục 4.1) cho macro precision **0,810** so với macro recall **0,690**. Một mô
hình mạnh hơn thì phần sai sót còn lại không còn là "chưa học được đặc trưng" mà chủ yếu là **quá
thận trọng với lớp hiếm** — đúng loại thiên lệch prior mà phép trừ `τ · log(prior)` được thiết kế để
sửa, và điều đó khớp với việc τ\* của `P2` lớn hơn và ổn định hơn. Trên `P1`, sai sót còn lẫn cả hai
loại nên không có một τ nào tốt cho mọi seed. Đây là **giả thuyết**; kiểm định nó cần đo τ\* trên
nhiều mức năng lực mô hình khác nhau, và chúng tôi chưa làm.

### 3.8 Bảng tổng kết mọi lever

| Lever | Tầng tác động | Δ macro-F1 | Giao thức | Kết luận |
|---|---|---|---|---|
| Chuẩn hoá ImageNet | đầu vào | — | bị ràng buộc bởi trọng số pretrained | giữ, không ablation |
| Lật ngang (`plain`) | augmentation | — (đường cơ sở) | 3 seed | **giữ** |
| Xoay + ColorJitter + Erasing | augmentation | −0,035 | 1 seed, cũ | **loại** (kèm chẩn đoán: tune sai LR) |
| **Công thức hiện đại trên CoAtNet @288** | huấn luyện | **+0,0443** | **3 seed** | **giữ** — lever **duy nhất** vượt ±0,035 |
| **Công thức hiện đại trên CoAtNet @224** (`P2c`) | huấn luyện | **+0,0358** | 3 vs 1 seed | **giữ** — ô lấp đầy bảng 2×2 |
| **Công thức hiện đại trên DenseNet @224** | huấn luyện | **−0,0110** | **3 seed** | **kết quả âm** — lever trên **không phổ quát** |
| **Tương tác công thức × kiến trúc** | huấn luyện × kiến trúc | **+0,0468** | 2×2, `top3` | **vượt ±0,035** — đây mới là thứ mua được điểm |
| Tương tác công thức × độ phân giải | huấn luyện × đầu vào | +0,0085 | 2×2, `top3` | **không phân giải được** |
| Balanced-Softmax | hàm mất mát | −0,007 / **−0,0047** | 1 seed × 2 lần chạy | **phẳng → loại** (lặp lại được) |
| cRT | classifier | −0,013 | 1 seed, cũ | **phẳng → loại** |
| Độ phân giải 224 → 288 | đầu vào | +0,0041 (công thức cũ) / **−0,0006** (công thức mới) | 3 seed / seed 0 | **không phân giải được**; triển khai nên dùng **224** |
| Đổi kiến trúc (DenseNet → CoAtNet) | kiến trúc | +0,0034 | **3 seed** | **không phân giải được** khi đứng một mình (mục 5.3) |
| **Pretrain IN-1k → IN-22k** (`A1`, Swin-T) | dữ liệu pretrain | **+0,0254** | 1 seed | lớn hơn **mọi** lever kiến trúc — hướng còn dư địa (mục 4.6) |
| **`top3` (ensemble checkpoint)** | cách đo | **+0,0188** TB | **3 seed, 4 mô hình** | **giữ** — 0 epoch, σ giảm ~2×, bền phần cứng |
| **Hiệu chỉnh logit trên `P2`** | logits | **+0,0143** | **3 seed** | **giữ** — 0 epoch, σ **co lại**, τ\* ổn định |
| Hiệu chỉnh logit trên `P1` | logits | +0,0052 | 3 seed | giữ nhưng **sát mép** — Δ < σ, một seed chọn τ\* = 0 |
| Lọc ảnh nghi rò rỉ | tập test | −0,0016 | 1 seed (seed 0) | kiểm chứng, không phải cải tiến |
| Ensemble 3 seed của `P2` | huấn luyện | → 0,7587 | 3 seed | **dòng riêng** (tốn 3 lần train) |
| Ensemble nhiều kiến trúc | huấn luyện | → 0,7130 | chọn trên val | **kết quả âm** — thấp hơn `P2` đơn lẻ |

---

## 4 · Nhãn & kiểm định — **20%**

### 4.1 Báo cáo per-class của mô hình đề xuất

`P2_coatnet0_288_modern`, seed 0, `top3` (bảng đầy đủ: `report/tables/18_per_class_va_confusion.txt`):

| | precision | recall | **F1** | support |
|---|---|---|---|---|
| **accuracy (= micro-F1)** | | | **0,850** | 1.586 |
| **macro avg** | **0,810** | **0,690** | **0,717** | 1.586 |
| weighted avg | 0,849 | 0,850 | 0,843 | 1.586 |

Đối chiếu đa chiều với bài báo (DenseNet-121): micro-F1 **0,850 vs 0,8203** (+0,030), macro-F1
**0,717 vs 0,6504** (+0,067 trên seed này).

**Khoảng cách precision–recall là con số đáng đọc nhất của bảng: 0,810 so với 0,690, chênh 0,120.**
Mô hình **thận trọng** với lớp hiếm: khi nó gọi tên một lớp hiếm thì thường đúng, nhưng nó bỏ sót
nhiều. Ba hệ quả, và cả ba đều được dùng ở chỗ khác trong báo cáo:

1. **Với một hệ thống sàng lọc, đây là chiều sai lệch kém mong muốn hơn** — bỏ sót bệnh lý tệ hơn
   báo động giả. Mục 9 ghi nó vào danh sách hạn chế.
2. **Nó giải thích vì sao hiệu chỉnh logit hoạt động trên `P2`** (mục 3.7): một mô hình thận trọng
   quá mức với lớp hiếm chính là mô hình mà phép dịch prior sửa được.
3. **Nó là bằng chứng thứ tư cho chẩn đoán "nút thắt là dữ liệu đuôi"** — mô hình không thiếu khả
   năng phân biệt, nó thiếu bằng chứng để dám quyết.

### 4.2 Ma trận nhầm lẫn

![Ma trận nhầm lẫn P2](figures/18_per_class_va_confusion.png)

> ⚠️ Hình đang tô theo **số đếm thô**, nên đường chéo của các lớp lớn áp hết dải màu và lỗi ở lớp
> hiếm gần như vô hình. Bản chuẩn hoá theo hàng sẽ đọc tốt hơn. Trong lúc chờ, **bảng 8 lớp yếu nhất
> dưới đây là cách đọc chính xác hơn** cho phần phân tích lỗi.

### 4.3 Tám lớp yếu nhất — và nút thắt thật sự

| Lớp | F1 | ảnh test | ảnh train |
|---|---|---|---|
| Mucosal inflammation large bowel | **0,286** | 6 | 17 |
| Cecum | 0,364 | 23 | 68 |
| Colorectal cancer | 0,435 | 28 | 83 |
| Resected polyps | 0,483 | 18 | 55 |
| Esophagitis | 0,485 | 21 | 64 |
| Colon diverticula | 0,500 | 6 | 17 |
| Barrett's esophagus | 0,600 | 19 | 57 |
| Gastric polyps | 0,636 | 13 | 39 |

Cột `ảnh train` là chỗ cần đọc: **cả 8 lớp yếu nhất đều nằm trong nhóm ít ảnh nhất** (17–83 ảnh
train, so với 760–880 của các lớp mạnh). Nút thắt là **biểu diễn đặc trưng (thiếu dữ liệu đuôi)**,
không phải hàm mất mát — khớp chính xác với kết luận độc lập ở mục 3.3.

### 4.4 Cải thiện nằm ở đâu — đối chiếu **Table 3 của bài báo**, từng lớp một

Việc kiểm chứng ở mục 1.2 trả về thứ giá trị hơn một con số headline: **Table 3 (trang 12)** liệt kê
precision/recall/F1 per-class của DenseNet-121 của họ, trên một tập test mà chúng tôi đã chứng minh là
cùng thành phần (mục 1.3). Nên phần cải thiện có thể **quy về từng lớp**, không phải khẳng định ở mức
tổng hợp. (22 giá trị F1 của Table 3 trung bình ra 0,6518, nhất quán với 0,6504 được công bố — nên
bảng đã được đọc đúng; `report/offline_tables.py` kiểm điều kiện đó và báo lỗi nếu lệch.)

Chia 22 lớp theo cỡ tập test (`tables-offline/32_per_class_vs_paper_table3.txt`):

| Nhóm | Số lớp | Δ F1 trung bình | Đóng góp vào macro-F1 |
|---|---|---|---|
| **Lớp hiếm** (< 50 ảnh test) | 15 | **+0,086** | **+0,0585 (90,4%)** |
| Lớp phổ biến (≥ 50 ảnh test) | 7 | +0,020 | +0,0062 (9,6%) |
| Toàn bộ 22 lớp | 22 | +0,065 | +0,0647 |

**90,4% mức cải thiện đến từ 15 lớp hiếm**, và năm mức tăng lớn nhất đều thuộc lớp có ≤ 21 ảnh test:

| Lớp | Paper F1 | `P2` F1 | Δ | test / train |
|---|---|---|---|---|
| Resected polyps | 0,17 | 0,483 | **+0,313** | 18 / 55 |
| Gastric polyps | 0,33 | 0,636 | **+0,306** | 13 / 39 |
| Retroflex rectum | 0,55 | 0,759 | **+0,209** | 13 / 40 |
| Barrett's esophagus | 0,40 | 0,600 | **+0,200** | 19 / 57 |
| Esophagitis | 0,31 | 0,485 | **+0,175** | 21 / 64 |

Năm lớp lớn nhất — Normal mucosa (294), Accessory tools (253), Normal stomach (194), Small bowel
(169), Colon polyps (164) — chỉ dịch +0,007 đến +0,046 và về cơ bản đã **bão hoà**: DenseNet-121 vốn
đã sát trần ở đó.

**Cách đóng khung có ý nghĩa lâm sàng:** các lớp hiếm ở đây là **bệnh lý** (Barrett's, esophagitis,
gastric polyps, resected polyps), còn các lớp phổ biến đã bão hoà là **giải phẫu bình thường**. Cải
thiện rơi vào đúng chỗ mà một hệ thống sàng lọc cần nó.

### 4.5 Nửa khó chịu của cùng bảng đó

Năm lớp **vẫn thua bài báo**, và phải báo cáo cả năm (nguyên tắc 5):

| Lớp | Δ so với paper | Kéo macro-F1 | ảnh test |
|---|---|---|---|
| **Mucosal inflammation large bowel** | **−0,214** | **−0,0097** | **6** |
| Colorectal cancer | −0,065 | −0,0030 | 28 |
| Blood in lumen | −0,059 | −0,0027 | 34 |
| Pylorus | −0,033 | −0,0015 | 79 |
| Duodenal bulb | −0,007 | −0,0003 | 41 |
| | | **tổng −0,0172** | |

Hai hệ quả, cả hai đều thuộc về báo cáo:

1. **+0,0937 là một ước lượng bảo thủ**, không phải một con số được tô hồng. Nó được báo *sau khi* đã
   hấp thụ −0,0172 từ năm lớp thua, trong đó −0,0097 chỉ từ một lớp 6 ảnh.
2. **Đây — không phải phương sai seed — là bất định chi phối.** Vì macro-F1 cân bằng 22 lớp như nhau,
   **một** lớp 6 ảnh mang trọng số 1/22 = 0,0455 trong con số cuối. Đoán đúng thêm 2 trong 6 ảnh của
   `Mucosal inflammation` sẽ đẩy macro-F1 lên ~+0,02: **lớn hơn toàn bộ khoảng cách Swin-T-vs-
   DenseNet-121, lớn hơn lever độ phân giải và lever kiến trúc cộng lại, và gấp ~2 lần σ của `P2`
   (0,0096)**. Đó chính là cơ chế cụ thể khiến bootstrap CI (±0,035) rộng gấp ~3,6 lần σ.

> **Vì vậy mua thêm seed không đáng.** `SEEDS = [0,1,2,3,4]` sẽ tinh chỉnh σ đi một chút, tốn ~4 giờ
> T4, trong khi để nguyên ±0,035 — bởi bootstrap lấy mẫu lại **tập test**, thứ mà `SPLIT_SEED = 42`
> giữ cố định cho mọi seed. Bất định còn lại là một vấn đề **dữ liệu** (2 trong 22 lớp có < 10 ảnh
> test), không phải vấn đề **ngẫu nhiên**, và không số lượng seed nào giải quyết được nó. Chính nhóm
> tác giả bộ dữ liệu đi đến cùng kết luận trong §4.3 của họ, khi đề xuất hướng few-shot cho đúng
> những lớp này.

**Một ghi nhận cho `P2`:** ở vòng 1, `Mucosal inflammation large bowel` có F1 = **0,000** và một mình
ngốn −0,023 macro-F1. `P2` đưa nó lên 0,286 — vẫn dưới 0,50 của bài báo, nhưng phần kéo giảm xuống
còn −0,0097. Đây là bằng chứng nữa cho việc công thức hiện đại **dịch chuyển đúng phần đuôi**, chứ
không chỉ nâng các lớp đã tốt.

*Lưu ý về phạm vi: so sánh per-class dùng seed 0, và F1 của một lớp 6 ảnh vốn không ổn định giữa các
seed; giá trị của bài báo làm tròn 2 chữ số thập phân. Phép chia tổng hợp (hiếm vs phổ biến) bền với
cả hai điều đó; từng dòng riêng của các lớp nhỏ nhất thì không.*

### 4.6 Dư địa còn lại nằm ở đâu — và vì sao đây là điểm dừng hợp lý

Mục 4.4 trả lời *"phần cải thiện đã qua đến từ đâu"*. Câu còn lại quan trọng hơn cho việc quyết định
có nên tiêu thêm giờ GPU: **phần chưa đạt nằm ở đâu?** Phép tính là một thí nghiệm tưởng tượng — đặt
F1 của một nhóm lớp bằng 1,0 rồi xem macro-F1 lên bao nhiêu. Nó **không phải dự báo**, chỉ là phân rã
số học khoảng cách đến trần 1,0 (`tables-offline/34_du_dia_con_lai.txt`):

macro-F1 hiện tại **0,7166** → còn **0,2834** đến trần lý thuyết.

| Nếu hoàn hảo ở… | macro-F1 | Đóng góp | % dư địa | Ảnh train / lớp | F1 hiện tại |
|---|---|---|---|---|---|
| **15 lớp hiếm** (< 50 ảnh test) | 0,9580 | **+0,2415** | **85,2%** | 73,5 | 0,646 |
| 7 lớp phổ biến (≥ 50 ảnh test) | 0,7585 | +0,0420 | 14,8% | 522,1 | 0,868 |
| *chỉ 2 lớp 6 ảnh test* | 0,7717 | *+0,0552* | | 17 | 0,393 |

**Ba hệ quả, và cả ba đều nói rằng dừng ở đây là hợp lý chứ không phải bỏ dở.**

1. **Riêng hai lớp 6 ảnh giữ +0,0552 — nhiều hơn cả đòn bẩy lớn nhất mà cả dự án đo được**
   (công thức hiện đại, +0,0443). Một cặp lớp có 17 ảnh train mỗi lớp đang nắm giữ nhiều macro-F1
   hơn toàn bộ công sức đổi công thức huấn luyện.
2. **Mọi lever nhắm vào *mô hình* đang tranh nhau 14,8% dư địa còn lại**, ở đúng nhóm lớp đã bão hoà
   (F1 0,868 trên 522 ảnh/lớp). 85,2% thì nằm ở chỗ mà chỉ **thêm dữ liệu** — pretrain trong domain,
   few-shot, hoặc thu thêm ảnh đuôi — mới chạm được.
3. **Cộng với ngưỡng phân giải ±0,035, điều này trở thành một chặn trên cứng.** Một lever nhắm vào mô
   hình phải mua được > 0,035 mới chứng minh được trên bộ test này — nhưng **trần của toàn bộ nhóm
   lớp phổ biến chỉ là +0,0420**. Nghĩa là kể cả một mô hình *hoàn hảo* trên 7 lớp phổ biến cũng chỉ
   vừa nhúc nhích qua ngưỡng. **Không còn lever nào thuộc họ "đổi mô hình" có thể chứng minh được
   trên bộ test 1.586 ảnh này**, bất kể bao nhiêu giờ GPU.

Đây là lý do chúng tôi chốt kết quả ở `P2` thay vì tiếp tục leo các bậc **nhắm vào mô hình**: với
họ lever đó thì **phép đo đã hết khả năng phân giải** — thêm bao nhiêu giờ GPU cũng không đổi được
kết luận này. Ngoại lệ duy nhất là `P4`: nó thêm **dữ liệu trong domain** chứ không đổi mô hình, nên
nằm ngoài lập luận trên, và được ghi là hướng phát triển tiếp (ngay dưới). Ba nguồn độc lập cùng chỉ
về một chỗ — bảng per-class (mục 4.3), khoảng cách precision–recall 0,120 (mục 4.1), và bảng dư địa
ở trên — và cả ba đều nói: phần còn lại là **vấn đề dữ liệu**, không phải vấn đề mô hình. Chính nhóm
tác giả bộ dữ liệu kết luận như vậy trong §4.3 của họ khi đề xuất hướng few-shot cho đúng những
lớp này.

#### Và có một số đo trực tiếp cho hướng đó, không chỉ một lập luận

Ablation `A1` đổi **duy nhất** bộ trọng số khởi tạo của Swin-T — ImageNet-1k → ImageNet-22k — giữ
nguyên kiến trúc, split, seed, số epoch, phần cứng (`tables-offline/36_*`):

| Thay đổi | seed 0, `top3` | Δ |
|---|---|---|
| `S0` Swin-T, pretrain **ImageNet-1k** | 0,6774 | — |
| `A1` Swin-T, pretrain **ImageNet-22k** | **0,7028** | **+0,0254** |
| *(để so)* `P0` − `B0`: đổi cả kiến trúc, cùng seed | 0,6718 vs 0,6878 | −0,0160 |

**Chỉ đổi dữ liệu pretrain mua được +0,0254 — nhiều hơn mọi lever kiến trúc mà dự án đo được, và
không đổi một dòng kiến trúc nào.** Đây biến luận điểm "hướng còn dư địa là dữ liệu" từ một suy luận
về phân bố lớp thành một **con số đã đo**. Nó cũng là cơ sở gần nhất mà chúng tôi có để kỳ vọng bậc
`P4` (pretrain trong domain trên HyperKvasir) sẽ dương — nhưng `A1` là ImageNet-22k, **không** phải
dữ liệu nội soi, nên con số +0,03…+0,08 dự kiến cho `P4` vẫn phải đọc là **phỏng đoán**, không phải
kết quả.

---

## 5 · Kiến trúc: CNN vs Transformer vs Hybrid — **15%**

### 5.1 Ba nhánh, một giao thức

| Vai trò | Mô hình | Trọng số | Tham số |
|---|---|---|---|
| **Baseline tham chiếu** (CNN, local) | DenseNet-121 | torchvision `IMAGENET1K_V1` | 7,0 M |
| **Baseline mới** (Transformer, global) | Swin-T | `timm swin_tiny_patch4_window7_224.ms_in1k` | 27,5 M |
| **Mô hình đề xuất** (Hybrid, conv + attention) | CoAtNet-0 | `timm coatnet_0_rw_224.sw_in1k` | 26,7 M |

**Vì sao hybrid là CoAtNet-0 chứ không phải "Swin-T + conv stem" tự ghép.** Phương án tự ghép nghe
sát whiteboard hơn, nhưng các lớp conv mới sẽ **khởi tạo ngẫu nhiên** trong khi phần Transformer đã
pretrained — nên phép so sẽ lẫn *kiến trúc* với *chất lượng khởi tạo*, đúng loại lỗi mà cả mục 3 và
mục 6 cho thấy là chi phối trên tập ~8k ảnh này. CoAtNet-0 là một hybrid **đã công bố và đã
pretrained trọn vẹn**, nên nó so được công bằng với hai baseline.

**Cả ba đều dùng trọng số ImageNet-1k.** Swin-T *có* bản in22k mạnh hơn và chúng tôi cố tình không
dùng nó ở dòng baseline: DenseNet-121 là in1k, đổi Swin sang in22k sẽ phá thế cân bằng và làm phép so
sánh CNN vs Transformer mất giá trị. Trọng số in22k được để dành thành một dòng ablation riêng —
⚠️ **nhưng ablation đó chưa từng chạy** (`RUN_ABLATIONS = False` trong notebook), nên đừng tìm con số
in22k ở bất kỳ bảng nào phía dưới.

Cùng split, cùng `SPLIT_SEED = 42`, cùng 3 seed `[0,1,2]`, AdamW 1e-4 / wd 1e-4, 30 epoch, batch 32,
**không tune riêng cho nhánh nào**. Chính sự cân bằng đó là thứ làm phép so sánh có giá trị. `P2` phá
sự cân bằng đó một cách có chủ ý (80 epoch, công thức khác) và vì vậy nó **không** nằm trong bảng so
sánh kiến trúc ở mục 5.3 — nó là một dòng riêng, ở mục 3.2.

### 5.2 Vì sao local-vs-global là câu hỏi thật với dữ liệu này

Ảnh nội soi mang **hai loại bằng chứng có cấu trúc rất khác nhau**, và hai họ kiến trúc bắt chúng
theo hai cách khác nhau:

* **Bằng chứng cục bộ, tần số cao.** Một polyp nhỏ, mép cắt sau can thiệp, một dụng cụ kim loại, kết
  cấu bề mặt niêm mạc — đây là các mẫu nhỏ, dịch chuyển bất biến. Convolution đúng là quy nạp
  (inductive bias) cho chuyện này: chia sẻ trọng số theo không gian + trường thụ cảm cục bộ, và nó
  học được từ ít dữ liệu.
* **Bằng chứng toàn cục, tầm xa.** *Vị trí* trong ống tiêu hoá — Cecum vs Retroflex rectum vs
  Ileocecal valve — không được xác định bởi một mảng nhỏ nào cả, mà bởi **bố cục toàn khung**: hình
  dạng lumen, hướng nếp gấp, quan hệ giữa các mốc. Đây đúng là thứ self-attention toàn cục
  (W-MSA/SW-MSA của Swin, bài 8) được thiết kế để mô hình hoá, còn CNN chỉ với tới sau nhiều lớp
  downsample.

Bằng chứng cho chiều thứ hai này **mỏng hơn ta muốn, và phải nói đúng như vậy**: trong các lớp mốc
vị trí, **chỉ Cecum thực sự yếu** — F1 0,364, lớp yếu thứ nhì của toàn bộ mô hình, dù có 23 ảnh test
(nhiều hơn vài lớp bệnh lý làm tốt hơn nó). Bốn mốc còn lại nằm ở mức trung bình trở lên
(Duodenal bulb 0,733 · Retroflex rectum 0,759 · Ileocecal valve 0,778 · GE junction 0,803), và
Retroflex rectum thậm chí là **lớp tăng mạnh thứ ba** so với bài báo (mục 4.4). Một lớp không đủ
thành một khuôn mẫu. Nên **giả thuyết "nút thắt là backbone, và một mô hình có attention sẽ phá nó"**
là một giả thuyết chính đáng để **đăng ký trước** — chứ không phải một điều bảng số đã chỉ ra sẵn.

**CoAtNet-0** là đích đến của whiteboard: conv stem ở các stage đầu (giữ quy nạp cục bộ, hiệu quả
dữ liệu) + attention ở các stage sau (bố cục toàn cục), tức lấy cả hai chứ không chọn một.

### 5.3 Kết quả — giả thuyết trên **không được số liệu ủng hộ**, và lần này có bằng chứng trực tiếp

Dưới `top3`, 3 seed (`tables/21_bang_tong_ket.txt`, `tables/17_bootstrap_ci.txt`):

| Mô hình | macro-F1 (3 seed) | so với paper 0,6504 | CI 95% (bootstrap, seed 0) |
|---|---|---|---|
| `B0_densenet121` (CNN) | 0,6780 ± 0,0073 | +0,0276 | [0,6412; 0,7239] — chưa kết luận được |
| `S0_swin_t` (Transformer) | 0,6813 ± 0,0081 | +0,0309 | [0,6307; 0,7167] — chưa kết luận được |
| `P0_coatnet0` (Hybrid @224) | 0,6814 ± 0,0100 | +0,0310 | [0,6229; 0,7103] — chưa kết luận được |
| `P1_coatnet0_288` (Hybrid @288) | 0,6855 ± 0,0068 | +0,0351 | [0,6364; 0,7185] — chưa kết luận được |
| — | — | — | — |
| **`P2_coatnet0_288_modern`** (+ công thức hiện đại) | **0,7298 ± 0,0096** | **+0,0794** | **[0,6660; 0,7531] — vượt** |
| **`P2c_coatnet0_224_modern`** (1 seed) | **0,7172** | **+0,0668** | **[0,6664; 0,7568] — vượt** |
| `P2b_densenet121_modern` | 0,6670 ± 0,0119 | +0,0166 | [0,6319; 0,7227] — chưa kết luận được |

**Bốn cấu hình của vòng 1 nằm trong dải 0,6780–0,6855 — trải rộng 0,0075, tức nhỏ hơn σ của chính
chúng.** Không cấu hình nào trong bốn có CI tách khỏi 0,6504. **Chỉ hai cấu hình đạt tiêu chuẩn mà
mục 1.4 đặt ra, và cả hai đều là CoAtNet-0 + công thức hiện đại** — một ở 288 px, một ở 224 px. Cấu
hình thứ ba dùng công thức ấy (`P2b`, trên DenseNet-121) thì không, nên tiêu chuẩn này phân biệt
đúng cái nó phải phân biệt: **tổ hợp**, chứ không phải riêng công thức.

Notebook có một **quy tắc quyết định đăng ký trước**: nếu `S0` chỉ *ngang* `B0` thì giả thuyết "nút
thắt là backbone" **không được ủng hộ**. Đó đúng là điều đã xảy ra, và lần này mạnh hơn vòng 1:

* `S0` − `B0` = **+0,0033**. `P0` − `B0` = **+0,0034**. Cả hai nhỏ hơn σ.
* Phép so theo cặp trên **từng seed** (`tables/17b_so_sanh_theo_cap.txt`) cho `S0`−`B0` dao động từ
  **−0,0104** (seed 0) đến **+0,0168** (seed 2), biên độ 0,0272 giữa các seed, và McNemar p ≈ 0,50
  ở cả ba. Cùng kết quả cho `P0`−`B0`.
* Và mục 2.5: **đổi phần cứng là đủ để đảo thứ tự xếp hạng** của bốn cấu hình này.

**Ở 3 seed, chúng tôi không được quyền nói Swin-T thắng DenseNet-121, cũng không được quyền nói
CoAtNet-0 thắng DenseNet-121.** Cách phát biểu đúng: *"ba kiến trúc không tách được nhau trên bộ test
này"* — yếu hơn và đúng hơn "hai bên ngang nhau", vì "ngang nhau" là một khẳng định về sự bằng nhau
mà dữ liệu cũng không chống đỡ được.

#### Vượt cả hai baseline — phép kiểm theo cặp

Bảng so sánh theo cặp của notebook (`tables/17b_*`) so **mọi thứ với `B0`**, nên nó không trả lời
được câu mà đề bài hỏi: *mô hình đề xuất có vượt **cả hai** baseline không?* Baseline 2 (`S0`) chưa
từng được kiểm theo cặp. Bổ sung ở `tables-offline/33_vs_hai_baseline.txt` (0 GPU, đọc từ logits đã
lưu, ghép cặp trên **cùng** bộ ảnh):

| `P2` vs | seed 0 | seed 1 | seed 2 | `ens3seed` | Kết luận |
|---|---|---|---|---|---|
| **`B0`** (baseline 1, = mô hình bài báo) | +0,0288 *chưa KL* | **+0,0632** ✅ | **+0,0635** ✅ | **+0,0430 [+0,0147; +0,0691]** ✅ | có ý nghĩa ở **2/3** seed |
| **`S0`** (baseline 2, Transformer) | **+0,0391** ✅ | **+0,0596** ✅ | **+0,0467** ✅ | **+0,0384 [+0,0074; +0,0708]** ✅ | có ý nghĩa ở **3/3** seed |

**Vượt baseline 2 còn sạch hơn vượt baseline 1**, và lý do không phải vì Swin-T yếu hơn DenseNet —
hai baseline chỉ cách nhau +0,0033 (dưới σ). Lý do là `B0` có **một seed may**: seed 0 của nó cho
0,7008 dưới quy tắc `best`, cao bất thường, đúng cái đã làm σ của nó phồng lên 0,0234 ở mục 1.4. Cái
may của riêng seed đó bị trừ vào phép so ở dòng seed 0. Đây là lần thứ hai `B0` seed 0 làm nhiễu một
phép so sánh — lần đầu ở vòng A100 (`../RESULTS.md` §10.8, phát hiện 2), và cách xử lý vẫn thế: **in
hết cả ba seed**, không chọn dòng đẹp.

⚠️ Cả hai phép so vẫn chỉ trên **một** bộ test cố định (`SPLIT_SEED = 42`). Ghép cặp làm phép so
nhạy hơn; nó **không** trả lời được *"kết quả có giữ trên bộ chia khác không"*.

#### Đường cộng dồn — mô hình đề xuất thắng bằng cách gì

Mô hình đề xuất thắng bằng cách **cộng dồn các lever đã được đo riêng**, không phải bằng một kiến
trúc mới bí ẩn. Đường cộng dồn khép kín chính xác, tất cả dưới quy tắc `top3`:

| Bước | Từ → đến | Δ |
|---|---|---|
| quy tắc chọn checkpoint | `B0` `best` 0,6686 → `B0` `top3` 0,6780 | +0,0094 |
| kiến trúc (CNN → hybrid CoAtNet-0) | `B0` 0,6780 → `P0` 0,6814 | +0,0034 |
| độ phân giải 224 → 288 | `P0` 0,6814 → `P1` 0,6855 | +0,0041 |
| **công thức huấn luyện hiện đại** | `P1` 0,6855 → `P2` **0,7298** | **+0,0443** |
| hiệu chỉnh logit (0 epoch) | `P2` 0,7298 → **0,7441** | +0,0143 |
| | **tổng** | **+0,0755** |

Cộng lại đúng bằng khoảng cách thật giữa hai đầu (0,7441 − 0,6686 = +0,0755), và +0,0937 so với
0,6504 của bài báo.

**Đây là bảng quan trọng nhất của báo cáo, vì nó cho thấy tiền đi đâu.** Một lever chiếm **59%** tổng
mức tăng và là lever duy nhất vượt ngưỡng phân giải ±0,035. Hai lever mà chúng tôi *đặt ra để kiểm
định* — kiến trúc và độ phân giải — cộng lại được +0,0075, tức **10%** tổng mức tăng và nằm trong
nhiễu. Hai lever "cách đo" cộng lại +0,0237 (**31%**) với **0 epoch phụ**.

**Một lưu ý để không cộng trùng:** lever quy tắc checkpoint ở đây phải lấy giá trị đo trên `B0`
(+0,0094), **không** phải giá trị đo trên `P1` (+0,0431) — con số sau đã bao gồm sẵn lợi thế của độ
phân giải 288, nên dùng nó sẽ đếm 288 hai lần. Dải +0,0014 … +0,0431 ở mục 3.5 là **biên độ của lever
qua bốn kiến trúc**, không phải một số hạng cộng được.

> ⚠️ **Lưu ý thứ hai, và nó quan trọng hơn: bảng này mô tả *con đường đã đi*, không phải một phân rã
> nhân quả.** Bậc "độ phân giải" được đo **dưới công thức cũ**; mục 3.4 cho thấy dưới công thức mới
> nó là **−0,0006**. Nên +0,0041 là số hạng *của đường đi này*, không phải một tài sản mang theo
> được — `P2c` bỏ hẳn bậc 288 mà vẫn tới 0,7172, và trên **cùng seed 0** còn nhỉnh hơn `P2` +0,0006.
>
> Phân rã nhân quả đúng nằm ở bảng 2×2 mục 3.2: **+0,0468 là số hạng tương tác `công thức × kiến
> trúc`**, và một số hạng tương tác thì **không tách được** thành hai lever cộng lại. Hai dòng
> "kiến trúc +0,0034" và "độ phân giải +0,0041" ở trên nhỏ đúng vì lý do đó: đo riêng lẻ, mỗi yếu tố
> gần như không mua được gì.

### 5.4 Đường học — 30 epoch đủ cho vòng 1, và **không** đủ cho công thức hiện đại

![Đường học val](figures/24_duong_hoc_val.png)

| Mô hình | val cao nhất | tại epoch | độ lệch 5 epoch cuối |
|---|---|---|---|
| B0_densenet121 | 0,6600 | **6** / 30 | 0,0069 |
| S0_swin_t | 0,6992 | **6** / 30 | 0,0178 |
| P0_coatnet0 | 0,6653 | 19 / 30 | 0,0133 |
| P1_coatnet0_288 | 0,6746 | 11 / 30 | 0,0290 |
| **P2_coatnet0_288_modern** | **0,7208** | **40** / 80 | **0,0019** |
| P2b_densenet121_modern | 0,6539 | 19 / 80 | 0,0002 |
| P2c_coatnet0_224_modern | 0,7086 | 11 / 80 | 0,0018 |

**Hai baseline đã học xong rất sớm** — cả hai đạt đỉnh val ở epoch 6/30, nên ngân sách 30 epoch là
thừa với chúng và 150 epoch của bài báo càng thừa. Đây là điều cho phép nói giao thức 30-epoch
**công bằng với baseline** (mục 1.4).

**`P2` đạt đỉnh ở epoch 40/80**, tức nửa đầu ngân sách của nó — nên 80 epoch là đủ, không phải thiếu.
Cột cuối cùng đáng đọc: độ lệch 5 epoch cuối của `P2` là **0,0019**, nhỏ nhất trong bốn cấu hình
3-seed và nhỏ hơn `P1` tới 15 lần. Đó là dấu vết của **EMA**: trung bình trượt trọng số làm phẳng
đuôi quá trình huấn luyện, và nó là lý do cơ học khiến `P2` có σ nhỏ dù chạy lâu hơn gấp gần 3 lần.

**`P2b` là dòng đáng chú ý ngược lại:** đỉnh val 0,6539 — **thấp hơn `B0` (0,6600)** dù được 80 epoch
thay vì 30. Không phải thiếu huấn luyện: nó đạt đỉnh ở epoch 19/80 rồi đi ngang 61 epoch với độ lệch
cuối 0,0002. Công thức hiện đại **không** giúp DenseNet-121, và val nói điều đó rõ hơn cả test.

**`P2c` bổ sung một chi tiết cho khuyến nghị ở mục 3.4:** cùng công thức hiện đại, ở 224 px nó đạt
đỉnh val ở **epoch 11/80** thay vì 40/80 của `P2` @288 — hội tụ sớm hơn gần 4 lần, độ lệch cuối
0,0018 tương đương. Nghĩa là 288 px không chỉ đắt hơn 1,70× mỗi ảnh lúc suy luận, nó còn **cần nhiều
epoch hơn để tới đỉnh** mà đỉnh thì không cao hơn.

### 5.5 Hai dòng ensemble — một dòng riêng, một kết quả âm

| | macro-F1 | CI 95% | Giá |
|---|---|---|---|
| Ensemble 3 seed của `P2` (+ hiệu chỉnh logit) | **0,7587** | [0,7110; 0,7924] | 3 lần huấn luyện |
| Ensemble nhiều kiến trúc (chọn tổ hợp **trên val**) | **0,7130** | — | 3 lần huấn luyện |

**Dòng dưới là một kết quả âm, và nó đã đảo chiều so với vòng 1.** Tổ hợp tốt nhất chọn trên val là
`S0 + P1 + P2`, cho test 0,7130 — **thấp hơn `P2` chạy một mình (0,7166 ở seed 0)**. Cơ chế rõ ràng:
ensemble trung bình xác suất, nên khi một thành viên vượt hẳn các thành viên khác, việc pha loãng nó
bằng hai mô hình yếu hơn 0,05 macro-F1 chỉ làm giảm điểm. Ở vòng 1, bốn cấu hình gần bằng nhau nên
ensemble giúp được (+0,03); ở vòng 2 điều kiện đó không còn.

> ⚠️ **Tổ hợp cao nhất trên tập *test* là `B0 + P2 + P2c` = 0,7358**, cao hơn tổ hợp chọn trên val
> **0,0228**. Dòng đó **không được báo cáo** như một kết quả: quét 127 tổ hợp rồi trích con số test
> cao nhất là chọn siêu tham số trên chính tập test, và con số thu được không còn là ước lượng không
> chệch. Notebook in ra cả hai dòng chính là để nói rõ điều đó — và khoảng cách val↔test **nới rộng**
> từ 0,0153 lên 0,0228 khi số tổ hợp tăng gấp đôi, đúng như lý thuyết về chọn lọc trên tập nhỏ dự
> đoán.

**Dòng trên vẫn là dòng riêng.** 0,7587 là con số cao nhất của dự án và nó rơi vào dải mục tiêu kỳ
vọng 0,72–0,75 của đề bài — nhưng nó tiêu **3 lần huấn luyện**, nên đặt nó cạnh một mô hình đơn rồi
gọi là "cải tiến kiến trúc" là không trung thực. Điều đáng nói là **mô hình đơn của chúng tôi cũng đã
vào dải đó**: `P2` + hiệu chỉnh logit = 0,7441 với **một** lần huấn luyện.

---

## 6 · Transfer learning: freeze vs trainable — **10%**

Bốn điều kiện trên **cùng một DenseNet-121**, cùng split, cùng seed, cùng ngân sách 30 epoch, cùng
chấm dưới `top3`. Chỉ **độ sâu được phép học** là khác.

| Điều kiện | Cái gì thực sự được huấn luyện | Seed | test macro-F1 | best VAL | phút/seed |
|---|---|---|---|---|---|
| **T1** linear probe | chỉ lớp phân loại | 1 | **0,5674** | 0,5409 | 20,0 |
| **T2** đóng băng nửa dưới | nửa trên + lớp phân loại | 1 | **0,6596** | 0,6397 | 20,2 |
| **T3** progressive unfreeze + LR phân biệt | probe 3 epoch → toàn mạng, LR backbone 0,5× | 1 | **0,6394** | 0,6166 | 23,5 |
| **T4** full fine-tune (= `B0`) | toàn mạng, một LR | 3 | **0,6780 ± 0,0073** | 0,6600 | 21,6 |

**Full fine-tune thắng, và cái giá tăng rất nhanh theo độ sâu đóng băng.** So với T4: T1 **−0,1106**,
T2 **−0,0184**, T3 **−0,0386** — cả ba vượt ngưỡng 2σ = **0,0146** (σ lấy từ 3 seed của `B0`, vì mỗi
điều kiện đóng băng chỉ có 1 seed). Khoảng cách không hề biên: linear probe **bỏ đi 11,1 điểm
macro-F1**, hơn toàn bộ những gì cả dự án này giành được so với baseline công bố (+9,4) — và đó là
phép so đáng suy nghĩ nhất của báo cáo.

**Vì sao, gói trong một câu:** đặc trưng ImageNet là đặc trưng của **ảnh tự nhiên**. Nội soi là một
modality khác — highlight phản chiếu, mặt nạ đen hình tròn, thống kê màu rất không tự nhiên — nên
chính **các lớp đầu** mới là lớp cần dịch chuyển, và đó đúng là phần mà đóng băng ghim chặt lại.

**T3 thua T2 (0,6394 vs 0,6596), và điều đó cần một câu.** Ở vòng 1 hai điều kiện này gần như bằng
nhau (0,6472 vs 0,6463); ở vòng T4 chúng cách nhau 0,0202, tức vượt 2σ. Nhưng **mỗi điều kiện chỉ có
1 seed**, nên phát biểu đúng là: *thứ tự T2/T3 không ổn định giữa hai lần đo*, và kết luận bền duy
nhất là **cả hai đều thua full fine-tune**. Đây chính là loại xếp hạng mà mục 2.5 cảnh báo không nên
tin.

**Bài báo xác nhận điều này miễn phí.** Table 2 của họ trộn cả hai chế độ (§4.2): chỉ fine-tune lớp
cuối cho 0,4496 / 0,4519 / 0,4883 (ResNet-152 / EfficientNet-B0 / DenseNet-169), fine-tune toàn mạng
cho 0,6176 / 0,6504 (ResNet-50 / DenseNet-121) — chênh **~0,16 trên đúng split này**, cùng chiều và
độ lớn khoảng 1,4 lần khoảng cách T1→T4 của chúng tôi. Hai thí nghiệm độc lập, một kết luận.

> **Ghi chú cài đặt xứng đáng một câu trong báo cáo:** một linear probe còn phải đưa **BatchNorm của
> phần đóng băng về chế độ `eval`**. `requires_grad = False` chặn gradient nhưng **không** chặn cập
> nhật running mean/var, nên một DenseNet-121 "đã đóng băng" theo cách ngây thơ (~120 lớp BN) vẫn
> tiếp tục trôi đặc trưng của chính nó qua từng epoch — và con số bạn đo được **không còn là linear
> probe nữa**. Hàm `_freeze_bn_of_frozen_part()` ở §11 của notebook xử lý việc này, và cùng lời gọi
> đó được tái dùng cho giai đoạn probe của T3.

**Ba giới hạn phải nói ra thay vì giấu.**

1. **T3 là LR phân biệt 2 nhóm** (backbone 0,5×, head 1×) sau 3 epoch probe — **không** phải
   layer-wise decay đúng nghĩa như ULMFiT/BEiT. Cách diễn đạt của đề bài cho phép cả hai, nhưng báo
   cáo không được nhận phương pháp mạnh hơn.
2. **T3 cũng không phải một thay đổi đơn biến so với T4.** Hàm `train_advanced` mà nó dùng còn kèm
   **cosine warmup 2 epoch** và **label smoothing 0,05**, hai thứ T1/T2/T4 không có. Nên khoảng cách
   T3 → T4 (−0,0386) là hiệu ứng của *cả gói*, không riêng lịch mở khoá. Theo chiều ngược lại,
   T1 được cho LR cao hơn (1e-3 thay vì 1e-4) vì nó chỉ còn một lớp linear phải học — dùng 1e-4 ở
   đó là tự làm yếu điều kiện rồi so sánh không công bằng.
3. **1 seed cho mỗi điều kiện đóng băng**, nên bảng này **xếp hạng thô**: chỉ những khoảng cách > 2σ
   ở trên mới được gọi là thật, và thứ tự T2/T3 thì không (xem đoạn trên). Nâng lên 3 seed tốn
   ~2 giờ T4 và là lever GPU rẻ nhất còn lại của hạng mục này.

---

## 7 · Deployment — *"Completeness of the Product"*

### 7.1 Export ONNX, độ trễ và kích thước — đo trên **T4**

| Mô hình | Độ phân giải | Tham số | ms/ảnh @ batch 1 | ms/ảnh @ batch 32 | Kích thước ONNX |
|---|---|---|---|---|---|
| DenseNet-121 | 224 | 7,0 M | 15,4 | **3,18** | 29,1 MB |
| Swin-T | 224 | 27,5 M | 19,5 | 4,82 | 113,7 MB |
| **CoAtNet-0 (`P2c`, nên triển khai)** | 224 | 26,7 M | **11,7** | 5,13 | 110,3 MB |
| **CoAtNet-0 (`P2`, được báo cáo)** | **288** | 26,7 M | 13,0 | 8,74 | 114,8 MB |

Cả bốn export ONNX thành công. **Hai cột xếp hạng các mô hình theo hai thứ tự khác nhau, và đó chính
là điểm cần nói.**

Ở **batch 1**, GPU rỗi gần hết wall-clock và phép đo bị chi phối bởi **số lần khởi chạy kernel**:
DenseNet-121 với ~120 lớp concat mất 15,4 ms dù chỉ 7 M tham số, **chậm hơn cả CoAtNet-0 @288**
(13,0 ms) vốn có 3,8× số tham số và 1,65× số pixel. Điều đó không khả thi về mặt vật lý nếu đọc như
chi phí tính toán.

Ở **batch 32**, các lần khởi chạy được khấu hao và các con số hành xử đúng: DenseNet-121 **nhanh
nhất** (3,18 ms, khớp với 7 M tham số của nó), và 288 tốn 8,74 / 5,13 = **1,70×** so với chính nó ở
224 — nằm trong sai số đo của tỉ lệ pixel **1,65×**, đúng như phải thế. Vòng A100 trước đó cho tỉ lệ
này là 1,68× (`tables-a100/28_*`); **hai GPU khác nhau cùng hội tụ về tỉ lệ pixel** là bằng chứng
mạnh nhất cho việc cột batch-32, không phải cột batch-1, mới là cột đo chi phí tính toán.

⚠️ **Không được trộn hai bảng.** Các con số tuyệt đối ở đây là của T4 và **thấp hơn 3–6 lần** so với
bảng A100 ở `tables-a100/28_*`; chỉ **tỉ lệ giữa các dòng** là so sánh được giữa hai phần cứng.

**Câu chi phí/lợi ích trung thực cho báo cáo:**

> Hệ thống đề xuất mua **+0,0937 macro-F1** so với baseline công bố, với giá **2,75× chi phí tính
> toán mỗi ảnh** (8,74 vs 3,18 ms ở batch 32), **3,9× kích thước mô hình** (114,8 vs 29,1 MB),
> **5,5× thời gian huấn luyện** (118,2 vs 21,6 phút/seed — tỉ lệ *epoch* chỉ là 2,7× nhưng mỗi epoch
> ở 288 px cũng nặng hơn), cộng thêm **hệ số 3×** ở suy luận cho ensemble top-3 checkpoint.
>
> Hạ về `P2c` @224 — cấu hình mục 3.4 khuyến nghị, chỉ kém −0,0006 trên seed 0 — đưa con số đầu
> xuống **1,61×** (5,13 vs 3,18 ms) và bỏ luôn 1,70× của độ phân giải. Đây là chỗ duy nhất trong báo
> cáo mà một lựa chọn **rẻ hơn** cũng **không tệ hơn**.

Cụm cuối cùng đó không được chôn vùi: `top3` chạy **3 checkpoint** cho mỗi ảnh. Mức tăng của quy tắc
chọn checkpoint **miễn phí ở lúc huấn luyện, không miễn phí ở lúc suy luận** — đó là hai khẳng định
khác nhau, và mục 3.5 chỉ được quyền nói khẳng định thứ nhất. Trong kịch bản lâm sàng một-ảnh-một,
chi phí là 3 × 13,0 ≈ **39 ms**, vẫn thoải mái real-time; trong sàng lọc theo lô là 3 × 8,74 ≈
**26 ms/ảnh** (với `P2c` là 3 × 5,13 ≈ **15 ms/ảnh**). Cả hai đều chấp nhận được, nhưng phải được **nói ra** thay vì để một bảng batch-1 hàm
ý.

*(Ghi chú: quy tắc đã chốt là `top3`, **không** `top3_tta`, nên hệ số suy luận là 3× chứ không phải
6×. Nếu chọn `top3_tta` — chỉ hơn khoảng 0,001 macro-F1 — chi phí suy luận sẽ tăng gấp đôi. Đó là một lý do
nữa để chốt `top3`, ngoài lý do xếp hạng ở mục 3.5.)*

### 7.2 Demo Gradio

Demo (§20b của notebook) nạp lại checkpoint đã lưu và phục vụ suy luận trên ảnh người dùng tải lên:

* **Đầu ra là top-5 kèm xác suất**, không phải một nhãn đơn. Với 22 lớp mất cân bằng và macro
  precision 0,810 / recall 0,690 (mục 4.1), một nhãn đơn là dạng đầu ra dễ gây hiểu sai nhất.
* **TTA lật ngang** ở lúc suy luận, khớp với augmentation lúc huấn luyện.
* **Đường suy luận được tách khỏi Gradio và tự kiểm tra trên một ảnh test thật trước khi dựng UI** —
  nên nó kiểm chứng được cả ở nơi không cài `gradio`. Log của vòng A100
  (`tables-a100/29_demo_gradio.txt`):

```
demo: da nap P1_coatnet0_288 seed 0 @ 288px tren cuda
  tu kiem tra: that = 'Accessory tools' | du doan = 'Accessory tools' (1.000)
```

> ⚠️ **Hai khoảng trống phải nêu, và cả hai đều là khoảng trống kỹ thuật đã biết, không phải lựa chọn
> thiết kế.**
>
> **1. Bằng chứng demo là của vòng A100, không phải vòng T4.** Nhánh resume của notebook cố tình
> **không** copy các file `.pt` (~100 MB mỗi file) từ nguồn chỉ-đọc về nơi ghi, nên ở vòng T4 ô demo
> báo *"không thấy `P1_coatnet0_288_seed0.pt`"* và bị bỏ qua. Đường suy luận không đổi giữa hai vòng,
> nhưng phát biểu đúng là *"demo đã được chạy thật ở vòng A100"*, không phải *"demo chạy được ở vòng
> đang báo cáo"*. Cách khép lại: đưa `P2_coatnet0_288_modern_seed0.pt` vào nguồn dữ liệu của phiên.
>
> **2. Demo yếu hơn hệ thống được báo cáo.** `run_seeds` chỉ lưu **một** checkpoint tốt nhất, nên
> demo chạy 1 checkpoint + TTA, **không phải** hệ thống `top3` + hiệu chỉnh logit (0,7441). Với `P2`,
> `best_tta` = **0,7199** so với 0,7441 — chênh 0,024. Khép lại khoảng trống này cần lưu cả 3 trạng
> thái (~1,3 GB) và huấn luyện lại; các lượt chạy cũ không thể bù ngược. Con số của **sản phẩm demo**
> vì vậy thấp hơn con số của **hệ thống được báo cáo**, và hai con số đó không được lẫn.

---

## 8 · Đối chiếu năm nguyên tắc bất di bất dịch của đề bài

| # | Nguyên tắc | Chúng tôi làm gì | |
|---|---|---|---|
| 1 | Dùng đúng split gốc; nếu tự chia thì công bố seed và giải thích | Bài báo không phát hành file split. Tự chia 60:20:20 phân tầng, `SPLIT_SEED = 42` công bố, **và kiểm chứng lại bằng Table 3 của họ**: 16/22 lớp khớp chính xác, 6 lớp lệch ±1, tổng trùng khít 1.586 | ✅ |
| 2 | Dữ liệu y tế → split theo bệnh nhân, và nói rõ đã kiểm tra | Bản phát hành chúng tôi tải về **chỉ gồm thư mục theo lớp + tên file dạng UUID/hash, không có định danh bệnh nhân nào**, nên split theo bệnh nhân là không khả thi từ dữ liệu công khai — và bài báo cũng chia 60:20:20 phân tầng theo ảnh, không theo bệnh nhân. Chúng tôi thay bằng **audit rò rỉ hai lớp** (MD5 + cosine ≥ 0,98) trên toàn bộ 7.930 ảnh, tìm được 9 cặp vắt qua các tập (6/1.586 ảnh test = 0,38%), rồi **tính lại macro-F1 sau khi bỏ chúng**: Δ = −0,0016 trên mô hình đề xuất | ⚠️→✅ đã kiểm tra và đo, kèm hạn chế nêu rõ |
| 3 | Dữ liệu mất cân bằng → luôn báo macro-F1 | macro-F1 là metric chính ở **mọi** bảng; accuracy/weighted-F1 chỉ để đối chiếu | ✅ |
| 4 | ≥ 3 seed, báo mean ± std | 6 cấu hình × 3 seed = 18 lượt, mean ± σ + bootstrap CI. **Ngoại lệ được nêu rõ:** `P2c` 1 seed (nên mọi phép so dùng nó đều đọc trên cùng seed 0), transfer learning 1 seed/điều kiện, ablation `A1`/`A2` 1 seed, và các ablation nhóm A 1 seed — tất cả đều đánh dấu và chỉ dùng ở mức kết luận | ✅ (kèm ngoại lệ được công bố) |
| 5 | Tự reproduce baseline mạnh nhất trước khi claim vượt; không được thì nói thẳng | Reproduce **0,6686 ± 0,0234** vs 0,6504 công bố (**+0,0182 = 0,78 σ**), dưới đúng quy tắc chọn checkpoint của họ, ở 30 thay vì 150 epoch. Một lần đo độc lập trước đó cho 0,6491 — **0,6504 nằm giữa hai lần đo của chúng tôi** (mục 1.4) | ✅ kèm nói thẳng rằng dải ±0,023 là rộng |

---

## 9 · Hạn chế

Tám điều chúng tôi biết là yếu, xếp theo mức độ ảnh hưởng tới kết luận:

1. **Bất định chi phối là dữ liệu, không phải mô hình.** Bootstrap CI ±0,035 rộng gấp ~3,6 lần σ giữa
   các seed (0,0096), vì 2 trong 22 lớp có < 10 ảnh test. Riêng lớp *Mucosal inflammation large
   bowel* (6 ảnh) kéo −0,0097 macro-F1. Không lượng seed nào chữa được; hướng đúng là few-shot hoặc
   thêm dữ liệu đuôi — trùng khuyến nghị của chính nhóm tác giả (§4.3).
2. **Số hạng tương tác — kết luận mạnh nhất của báo cáo — tựa lên một ô chỉ có 1 seed.** `P2c` lấp
   được ô trống của bảng 2×2 và cho +0,0468 / +0,0497, vượt ngưỡng ±0,035; nhưng nó mới chạy **1
   seed**, nên chúng tôi chưa có σ cho chính số hạng đó. Hai phép so an toàn nhất (mục 3.2 mục B)
   đều được đọc **trên cùng seed 0** để không trộn 1 seed với 3 seed. Nâng `P2c` lên 3 seed tốn
   ~3,3 giờ T4 và là hạng mục còn thiếu **quan trọng nhất**.
3. **Khuyến nghị triển khai `P2c` @224 dựa trên một khoảng cách không phân giải được.** −0,0006 giữa
   224 và 288 dưới công thức mới nghĩa là *"không đo thấy khác biệt"*, **không** phải *"đã chứng minh
   bằng nhau"* — cùng loại phát biểu mà mục 5.3 từ chối đưa ra cho ba kiến trúc. Khuyến nghị này
   đứng vững vì phía chi phí thì **đo được chắc chắn** (1,70×), chứ không vì phía độ chính xác đã
   được chứng minh.
4. **So sánh kiến trúc không phân giải được ở 3 seed, và mục 2.5 cho thấy nó còn không bền qua một
   lần đổi phần cứng.** Chúng tôi phát biểu là *"không tách được"* — nhưng điều đó cũng có nghĩa là
   **luận điểm local-vs-global ở mục 5.2 vẫn là một giả thuyết chưa được kiểm định**, không phải một
   kết luận đã đo. Đây là hạn chế trực tiếp của hạng mục 15%.
5. **Trong ba ablation cân bằng lớp, mới một cái được đo lại dưới giao thức hiện hành.**
   Balanced-Softmax đã có `A2` (T4, `top3`, cùng split/seed/epoch với `B0`): −0,0047, cùng dấu và
   cùng độ lớn với −0,007 của nhóm A. **cRT thì chưa** — nó vẫn là 1 seed, giao thức cũ, trọng số
   không còn tồn tại, nên con số −0,013 chỉ được trích như **kết luận** ("phẳng"), không như số đo.
6. **Bốn cấu hình vòng 1 đã bị huấn luyện lại ngoài ý muốn** khi chuyển hạ tầng, nên bản báo cáo
   trước của chúng tôi trích một bộ số mà `.npz` nền không còn (`../RESULTS.md` §10.9). Toàn bộ báo
   cáo này đã được đặt lại trên vòng T4. Chúng tôi giữ lỗi này trong danh sách hạn chế thay vì xoá
   dấu vết, vì nó là nguồn của phát hiện ở mục 2.5 — và vì một quy trình để lọt được lỗi đó một lần
   thì có thể để lọt lần nữa. Biện pháp đã thêm: kiểm dòng *"đọc lại từ `.npz`"* trong bảng đầu tiên
   sau mỗi phiên.
> **Phiên `SESSION = 4` (T4, 5,6 giờ) đã lấp ba hạn chế của bản trước** — ô trống của bảng 2×2
> (`P2c`), `P2b` đủ 3 seed, và hai ablation `A1`/`A2` — và đúng như dự kiến, nó **không nâng
> macro-F1**: con số báo cáo vẫn là 0,7441. Nó đổi ba câu *"chưa kết luận được"* thành ba câu có
> bằng chứng, và sinh ra hai hạn chế mới (mục 2 và 3 ở trên) hẹp hơn hẳn ba cái nó thay thế.
> Chi tiết: `../RESULTS.md` §10.10.

7. **Sản phẩm demo yếu hơn hệ thống được báo cáo** (`best_tta` 0,7199 so với 0,7441), và bằng chứng
   demo hiện là của vòng A100 (mục 7.2). **Không có ID bệnh nhân**, nên không loại trừ được khả năng
   nhiều frame của cùng một ca nằm khác tập; audit cosine là một xấp xỉ, và 9 cặp nó tìm ra là **chặn
   dưới**, không phải chặn trên.
8. **Một bảng trong notebook đã được in ra dưới sai quy tắc chấm điểm.** Ô §15c đọc `SELECTION_RULE`,
   nhưng biến đó chỉ được **chốt ở §16** — chạy sau nó — nên phiên `SESSION = 4` in bảng 2×2 dưới
   `best` thay vì `top3`, và dòng kết luận đi kèm đọc ngược dấu ở thành phần độ phân giải. Bảng đúng
   được tính lại ở `tables-offline/35_*` và mọi số trong báo cáo lấy từ đó. Nguyên nhân gốc đã được
   sửa (§15c–§15f và §16 giờ gọi **chung** một hàm `vote_rule()`), nhưng output cũ trong `.ipynb` thì
   không sửa lại được nếu không chạy lại GPU. Đây là lần **thứ hai** cùng một loại lỗi xuất hiện —
   lần đầu là ghim cứng tên quy tắc (`../RESULTS.md` §10.9) — và cả hai lần đều bị bắt bởi cùng một
   thứ: bắt mọi con số phải khai nó sinh ra từ bảng nào.

---

## 10 · Kết luận

Chúng tôi vượt baseline công bố của GastroVision — **0,7441 ± 0,0088 so với 0,6504, với CI 95% không
chứa con số đó** — và điều đáng nói nằm ở **cách** vượt, vì nó không phải cách chúng tôi dự định.

**Câu trả lời không phải một kiến trúc mới, và lần này chúng tôi có số để nói điều đó chắc chắn.** Khi
phân rã toàn bộ +0,0755 thành các thành phần đo riêng dưới một quy tắc duy nhất, hai lever mà cả dự
án được thiết kế để kiểm định — **đổi kiến trúc (+0,0034) và tăng độ phân giải (+0,0041)** — cộng lại
được 10% mức tăng và nằm gọn trong nhiễu. Thứ trả tiền là **công thức huấn luyện** (+0,0443, 59%, và
là lever *duy nhất* vượt ngưỡng phân giải ±0,035 của bộ test này) và **cách đo** (+0,0237, 31%, với
0 epoch phụ).

Nhưng ngay cả kết luận đó cũng có điều kiện, và điều kiện ấy là kết quả âm quan trọng nhất của báo
cáo: **cùng công thức hiện đại đó làm DenseNet-121 tệ đi** — **−0,0110**, cả 3/3 seed đều âm, và
thấp hơn cả trên val. Bản trước của báo cáo dừng ở đó và phải viết *công thức × (kiến trúc **hoặc**
độ phân giải)* vì hai phép so không chia sẻ độ phân giải. Ô thứ tư của bảng 2×2 (`P2c` = CoAtNet-0
@224 + cùng công thức) đã được chạy và trả lời dứt khoát:

> **công thức × kiến trúc = +0,0468 · công thức × độ phân giải = +0,0085.** Ở cùng seed 0, mỗi yếu
> tố đứng một mình đều ≤ 0 (kiến trúc −0,0160; công thức −0,0043) còn ghép lại thì +0,0294 — nên
> **thứ mua được điểm là một số hạng tương tác, không phải một lever cộng thêm.**

Kèm theo là kết quả âm thứ hai, và nó lật một phỏng đoán ngầm của chính chúng tôi: **288 px không
phải thứ làm nên chuyện.** Dưới công thức mới, 288 so với 224 cho **−0,0006** trong khi tốn **1,70×**
chi phí tính toán mỗi ảnh — nên cấu hình *nên triển khai* là `P2c` @224, dù con số *được báo cáo* vẫn
là `P2` vì nó có đủ 3 seed. Đây là chỗ báo cáo này khác một báo cáo chỉ đi tìm con số cao nhất: hai
dòng phủ định hai giả thuyết đẹp nhất của chúng tôi đều được giữ nguyên, và ô thí nghiệm còn trống
thì được chạy chứ không được ghi chú rồi bỏ qua.

**Nền nhiễu là nhân vật chính.** Trên một tập ~8.000 ảnh với 22 lớp mất cân bằng 50,6×, hai lớp có
dưới 10 ảnh test, nên bootstrap CI rộng ±0,035 — **lớn hơn bốn trong năm lever mà chúng tôi đo**.
Phần lớn công việc kỹ thuật thực sự là hạ nền nhiễu đó xuống: ensemble top-3 checkpoint giảm σ trung
bình gần một nửa với 0 epoch phụ, dương trên cả năm kiến trúc, và — theo một phép lặp lại tình cờ có
được trên hai loại GPU — **bền với phần cứng gấp 4 lần** các quy tắc một-checkpoint. Cùng phép lặp
lại đó cho thấy **xếp hạng kiến trúc của vòng 1 bị đảo chỉ bởi việc đổi GPU**. Một xếp hạng như vậy
không phải một xếp hạng, và chúng tôi báo cáo nó như thế.

Ba lever mà sách vở khuyến nghị cho dữ liệu đuôi dài **lúc huấn luyện** — augmentation mạnh,
Balanced-Softmax, cRT — đều phẳng hoặc hồi quy; Balanced-Softmax nay đã được đo lại dưới giao thức
hiện hành (`A2`: −0,0047) và **lặp lại đúng kết quả âm cũ**. Đó không phải một chỗ tắc; nó là một chẩn đoán, và
chẩn đoán ấy được xác nhận từ **bốn** chiều độc lập: 8 lớp yếu nhất chính là 8 lớp ít ảnh nhất; 90,4%
mức cải thiện so với bài báo đến từ 15 lớp hiếm; macro precision (0,810) vượt macro recall (0,690)
tới 0,120; và đóng băng backbone — tức từ chối cho đặc trưng dịch chuyển — làm mất 11,1 điểm macro-F1,
nhiều hơn toàn bộ mức cải thiện của cả dự án. Nút thắt là **biểu diễn đặc trưng**, không phải hàm mất
mát.

Với một ngoại lệ đáng giá: **can thiệp mất cân bằng lúc *suy luận* thì có hiệu quả** (+0,0143 từ hiệu
chỉnh logit, tái lập được trên 3 seed). Phép so sạch nhất cho điều đó nằm trên **cùng một backbone,
cùng seed, cùng số epoch**: chữa trong hàm mất mát (`A2`) cho **−0,0047**, chữa lúc suy luận cho
**+0,0213** — cùng mục tiêu, hai dấu ngược nhau, và bên thắng tốn 0 epoch. Và câu chuyện của lever này là bài học phương pháp mà chúng
tôi mang đi: chính lever đó bị **loại** ở vòng 1 vì đo trên `P1` thì nó không tái lập. Kết luận
*"không có gì để ăn ở đây"* hoá ra là một phát biểu về **một mô hình**, không phải về **bộ dữ liệu** —
và chúng tôi đã suýt dùng nó để cắt cả một nhánh của kế hoạch.

Về mặt lâm sàng, cải thiện rơi vào đúng chỗ cần: các lớp hiếm ở đây là **bệnh lý**, các lớp phổ biến
đã bão hoà là **giải phẫu bình thường**. Và +0,0937 được báo *sau khi* đã hấp thụ −0,0172 từ năm lớp
vẫn thua bài báo, nên nó là ước lượng bảo thủ.

**Và đây là điểm dừng, không phải chỗ bỏ dở.** Mục 4.6 phân rã khoảng cách còn lại đến trần: **85,2%
nằm ở 15 lớp hiếm** (73,5 ảnh train mỗi lớp), còn trần của **toàn bộ** nhóm 7 lớp phổ biến chỉ là
**+0,0420** — tức một mô hình *hoàn hảo* trên nhóm đã bão hoà cũng chỉ vừa nhúc nhích qua ngưỡng phân
giải ±0,035 của bộ test này. Nói cách khác: **không còn lever nào thuộc họ "đổi mô hình" có thể chứng
minh được trên bộ test 1.586 ảnh này, bất kể bao nhiêu giờ GPU.** Riêng hai lớp 6 ảnh giữ +0,0552 —
nhiều hơn đòn bẩy lớn nhất cả dự án đo được. Hướng còn lại có dư địa là **thêm dữ liệu trong domain**
(pretrain trên bộ nội soi khác, hoặc few-shot cho đúng các lớp đó), không phải một backbone khác —
và điều đó giờ có một **số đo** đứng sau chứ không chỉ một lập luận: đổi *duy nhất* bộ trọng số
pretrain của Swin-T từ ImageNet-1k sang ImageNet-22k mua được **+0,0254**, nhiều hơn mọi lever kiến
trúc mà dự án này đo được (mục 4.6).

Nếu phải rút một câu mang đi: **trên dữ liệu y tế quy mô nhỏ, hãy đo cho đủ chắc trước khi tin bất kỳ
lever nào — vì phần lớn lever nhỏ hơn sai số của chính phép đo, và cái sống sót lại không phải cái
ta đi tìm.**

---

## Tài liệu tham khảo

1. **GastroVision** — Jha, D. và cộng sự. *GastroVision: A Multi-class Endoscopy Image Dataset for
   Computer Aided Gastrointestinal Disease Detection.* ICML Workshop on Machine Learning for
   Multimodal Healthcare Data, 2023. arXiv:2307.08140.
   Dữ liệu: [GitHub](https://github.com/DebeshJha/GastroVision) · [OSF](https://osf.io/84e7f/)
2. **DenseNet** — Huang, G. và cộng sự. *Densely Connected Convolutional Networks.* CVPR 2017.
   arXiv:1608.06993.
3. **Swin Transformer** — Liu, Z. và cộng sự. *Swin Transformer: Hierarchical Vision Transformer
   using Shifted Windows.* ICCV 2021. arXiv:2103.14030.
4. **CoAtNet** — Dai, Z. và cộng sự. *CoAtNet: Marrying Convolution and Attention for All Data
   Sizes.* NeurIPS 2021. arXiv:2106.04803.
5. **Logit adjustment** — Menon, A. K. và cộng sự. *Long-tail learning via logit adjustment.*
   ICLR 2021. arXiv:2007.07314.
6. **timm** — Wightman, R. *PyTorch Image Models.* github.com/huggingface/pytorch-image-models.
7. Đề bài môn học: *9 đề bài Deep Learning — chọn dataset có baseline công bố & vượt qua nó*
   (`mse-dl-de-bai-vuot-baseline.pdf`), mục 6 "Khung report bám đúng tỉ lệ 70/30".

---

## Phụ lục · Nơi mỗi con số trong báo cáo này được sinh ra

Không con số nào được gõ tay. Toàn bộ được trích tự động từ output đã lưu của notebook bằng
`report/extract.py`, hoặc tính lại từ logits đã lưu bằng `report/offline_tables.py`.

**Vòng chạy:** Kaggle, profile `gpu-t4` (Tesla T4), `SEEDS = [0,1,2]`, `SPLIT_SEED = 42`, batch 32,
AMP float16 + GradScaler, `SELECTION_RULE = "top3"`. 30 epoch cho `B0`/`S0`/`P0`/`P1` và cho hai
ablation `A1`/`A2`; **80 epoch** cho `P2`/`P2b`/`P2c` với công thức hiện đại.

> ⚠️ **macro-F1 KHÔNG bất biến với phần cứng khi phải huấn luyện lại.** Nó chỉ bất biến khi thực sự
> nạp lại `.npz` rồi tính lại từ logits đã lưu. Một bản trước của phụ lục này khẳng định điều ngược
> lại, và chính khẳng định đó là lý do việc huấn luyện lại ở vòng 2 đi lọt. Mức lệch đo được:
> `tables-offline/30_lap_lai_a100_vs_t4.txt`; câu chuyện đầy đủ: `../RESULTS.md` §10.9.

**Ba thư mục nguồn, đừng lẫn:**

| Thư mục | Là gì |
|---|---|
| `tables/`, `figures/` | Vòng T4 hiện tại — mặc định cho mọi con số |
| `tables-a100/`, `figures-a100/` | Vòng A100 27-08-2026. Là **nguồn duy nhất** còn lại của Gate 0a (mục 2.5), demo (mục 7.2) và bảng độ trễ A100 |
| `tables-offline/` | Tính lại 0 GPU — các bảng không ô nào của notebook in ra được. Bảng 30–34 từ `../ckpt-t4/*.npz`; bảng 35–36 đọc lại chính `tables/*.txt` và **đối chiếu với bảng pivot mục 16 trước khi ghi** |

| Mục của báo cáo | File nguồn trong `report/` |
|---|---|
| 1.3 giao thức | `tables/02_ho_so_chay.txt` |
| 1.4 tái lập baseline | `tables/12_B0_densenet121.txt`, `tables/16_bang_6_quy_tac.txt` |
| 2.1 lọc lớp, split | `tables/04_loc_lop_22.txt`, `tables/05_chia_split.txt` |
| 2.2 phân bố lớp | `tables/06_eda.txt`, `figures/06_eda.png` |
| 2.3–2.4 audit rò rỉ & nhãn | `tables/07_audit_md5.txt`, `tables/08_audit_gan_trung.txt` |
| 2.5 Gate 0a | `tables-a100/11_gate0a_tat_dinh.txt` |
| **2.5 lặp lại A100 ↔ T4** | **`tables-offline/30_lap_lai_a100_vs_t4.txt`** |
| **3.2 công thức hiện đại, bảng 2×2, tương tác** | **`tables/15b_P2_cong_thuc_hien_dai.txt`, `tables-offline/35_bang_2x2_tuong_tac.txt`** |
| **3.3 / 3.7 mất cân bằng: hàm mất mát vs suy luận · 4.6 dữ liệu pretrain** | **`tables-offline/36_mat_can_bang_va_pretrain.txt`**, `tables/25_ablation_tuy_chon.txt` |
| **3.4 độ phân giải dưới hai công thức** | **`tables-offline/35_bang_2x2_tuong_tac.txt`** (mục C) |
| 3.5 sáu quy tắc checkpoint | `tables/16_bang_6_quy_tac.txt` |
| 3.6 lọc rò rỉ | `tables/22_do_ben_truoc_ro_ri.txt` |
| **3.7 hiệu chỉnh logit** | `tables/19_donbay_hieuchinh_logit.txt`, **`tables-offline/31_he_thong_p2_hieu_chinh_logit.txt`** |
| 4.1–4.3 per-class | `tables/18_per_class_va_confusion.txt`, `figures/18_per_class_va_confusion.png` |
| **4.4–4.5 đối chiếu Table 3** | **`tables-offline/32_per_class_vs_paper_table3.txt`** |
| **4.6 dư địa còn lại** | **`tables-offline/34_du_dia_con_lai.txt`** |
| 5.3 bảng tổng kết + CI | `tables/21_bang_tong_ket.txt`, `tables/17_bootstrap_ci.txt` |
| **5.3 so sánh theo cặp (bậc M)** | **`tables/17b_so_sanh_theo_cap.txt`** |
| 5.4 đường học | `tables/24_duong_hoc_val.txt`, `figures/24_duong_hoc_val.png` |
| 5.5 ensemble | `tables/20_donbay_ensemble_kientruc.txt`, `tables-offline/31_*` |
| 6 transfer learning | `tables/27_transfer_learning.txt`, `tables/26_transfer_learning_log.txt` |
| 7.1 ONNX & độ trễ | `tables/28_trien_khai_onnx_do_tre.txt` (T4), `tables-a100/28_*` (A100) |
| 7.2 demo | `tables-a100/29_demo_gradio.txt` |
| Ablation nhóm A (1 seed, giao thức cũ) | `../RESULTS.md` §2, §6 |

Notebook: `../notebooks/final-gastrovision-classification.ipynb` (sinh ra từ `../build_notebook.py` — đừng
sửa tay file `.ipynb`). Nhật ký đo đầy đủ, gồm cả các nhánh đã bị loại: `../RESULTS.md`.
