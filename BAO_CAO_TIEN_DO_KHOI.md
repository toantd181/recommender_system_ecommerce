# Báo cáo tiến độ – phần việc của Khôi

*Ngày 23/11/2026 · Hệ gợi ý lai cho sàn TMĐT (IT4613) · Dữ liệu: REES46, mẫu 5% người dùng*

## 1. Tóm tắt

- Đã hoàn thành cả **4 việc** được giao ở mục 6 của `BAO_CAO_TIEN_DO.md`: khung đánh giá và baseline, FP-Growth "thường được mua cùng", gợi ý cho khách vãng lai, demo Streamlit.
- **Kết quả tốt nhất trên tuần test (24–30/11), kèm khoảng tin cậy 95%:**

  | Mô-đun | Mô hình | HitRate@10 | Baseline tốt nhất |
  |---|---|---|---|
  | Khách vãng lai (đoán sản phẩm tiếp theo trong phiên) | **Bộ xếp hạng LightGBM (LTR)** | **0,345** [0,339; 0,350] | Co-visitation: 0,262 |
  | Thường được mua cùng | **FP-Growth + fallback** | **0,308** [0,290; 0,327] | Phổ biến cùng danh mục: 0,247 |
  | Gợi ý Top-K cho người dùng | Chỉ có baseline (ALS, content-based: phần của Dương) | 0,187 [0,180; 0,195] | — |

- Mọi cải thiện trên đều có ý nghĩa thống kê theo kiểm định bootstrap cặp (p < 0,001).
- Khung đánh giá đã **khớp với slide L8**: tính đủ Precision, Recall, F1, HitRate, MRR, MAP, nDCG, Rank Score, Liftindex. Mỗi metric được kiểm tra bằng đúng ví dụ số trong slide.
- 12/12 test qua. Demo chạy ở `http://localhost:8501`.

## 2. Dữ liệu

| Bước | Kết quả |
|---|---|
| Lấy mẫu | Lấy 5% người dùng theo hash `user_id`, đọc stream trực tiếp từ `archive.zip` (không cần giải nén 14,7 GB) |
| Tháng 11 | 3,39 triệu sự kiện → `data/nov_u5.parquet` |
| Tháng 10 | Cùng tập người dùng, 2,13 triệu sự kiện → `data/oct_u5.parquet` |
| Gộp 10+11 | 5,52 triệu sự kiện → `data/octnov_u5.parquet` |
| Chia thời gian | Lịch sử đến 16/11 · validation 17–23/11 · test 24–30/11 |

- **Lấy mẫu theo người dùng tốt hơn lấy mẫu theo dòng:** mẫu 5% người dùng có 3.312 phiên mua có ≥2 sản phẩm, trong khi mẫu 10% theo dòng (báo cáo cũ) chỉ có 1.081.
- **Mức trùng người dùng giữa tháng 10 và 11** (báo cáo cũ ghi là chưa kiểm tra): trong số người dùng có thêm giỏ/mua ở tuần test, tỉ lệ đã có lịch sử tăng từ 66,4% (chỉ tháng 11) lên 73,6% (tháng 10+11).

## 3. Khung đánh giá (`src/evalkit.py`)

- **Nhãn dương:** thêm giỏ hoặc mua (chỉ dùng mua thì quá thưa).
- **Báo cáo theo nhóm người dùng:**
  - `unseen`: chưa có lịch sử;
  - `light`: 1–4 tương tác;
  - `heavy`: từ 5 tương tác trở lên.
- **Metric theo slide L8:**

  | Metric | Slide | Ghi chú |
  |---|---|---|
  | Precision@K, F1@K | 9, 11 | |
  | Recall@K | 12 | Chia cho `min(P_u, K)` theo định nghĩa trong slide |
  | MRR | 17 | |
  | Rank Score (α = 2), Liftindex | 19–20 | |
  | nDCG | 21 | IDCG theo `min(\|T\|, K)` (cách chuẩn); slide tính theo số lượt trúng, hai cách cho cùng kết quả trên ví dụ trong slide |
  | MAP | 24–25 | |

- **Chọn tham số trên validation, test chỉ chạy một lần**, để kết quả không bị lạc quan do dò tham số trên chính tập test.
- **Khoảng tin cậy 95%:** dùng cluster bootstrap, lấy mẫu lại theo phiên hoặc giỏ vì các truy vấn trong cùng một phiên tương quan với nhau. So sánh hai mô hình bằng bootstrap cặp.
- **Metric ngoài độ chính xác:**
  - độ phủ catalog;
  - độ mới lạ;
  - tỉ lệ gợi ý thuộc 1% sản phẩm hot nhất.
- **Không áp dụng:**
  - MAE/RMSE (slide 26): không có điểm đánh giá.
  - AUC (slide 13–16): đây là bài toán Top-K, không phải phân loại có ngưỡng.
  - A/B testing (slide 30): cần người dùng thật.

## 4. Kết quả chi tiết (tuần test)

### 4.1 Gợi ý Top-K cho người dùng (baseline, 10.101 người dùng)

| Mô hình | Precision | Recall | F1 | MRR | MAP | nDCG | Rank Score | Liftindex | HitRate |
|---|---|---|---|---|---|---|---|---|---|
| PopularRecent | 0,019 | 0,137 | 0,033 | 0,072 | 0,070 | 0,080 | 0,067 | 0,113 | 0,172 |
| PopularByCategory | 0,021 | 0,151 | 0,035 | 0,087 | 0,085 | 0,093 | 0,081 | 0,129 | 0,187 |

- Precision thấp là bình thường: mỗi người chỉ có khoảng 1–2 sản phẩm đúng trong tuần test.
- ALS và content-based của Dương cần vượt `PopularByCategory`.

### 4.2 Khách vãng lai – đoán sản phẩm tiếp theo (20.000 phiên, 89.460 truy vấn)

| Hệ thống | HitRate@10 | MRR@10 |
|---|---|---|
| Phổ biến | 0,035 | 0,012 |
| Item2Vec | 0,157 | 0,074 |
| Item2Vec + thuộc tính (danh mục, thương hiệu) | 0,210 | 0,101 |
| Co-visitation | 0,259 | 0,103 |
| Co-visitation có trọng số thời gian | 0,262 | 0,104 |
| Lai: co-visitation → Item2Vec | 0,276 | 0,106 |
| Lai + 3 sản phẩm vừa xem lên đầu (quy tắc đơn giản) | 0,315 | 0,156 |
| **Bộ xếp hạng LightGBM (LTR)** | **0,345** | **0,165** |

**Bộ xếp hạng (`src/ltr_session.py`):**
- Ứng viên lấy từ co-visitation, Item2Vec và các sản phẩm đã xem trước đó trong phiên.
- 14 đặc trưng, gồm: điểm co-visitation, độ tương đồng Item2Vec, số bước kể từ lần xem trước, cùng danh mục/thương hiệu với sản phẩm vừa xem, tỉ lệ giá, độ phổ biến, tỉ lệ thêm giỏ.
- Huấn luyện bằng LambdaRank trên tuần validation.

### 4.3 Thường được mua cùng (1.833 giỏ test, 4.242 truy vấn)

| Hệ thống | HitRate@10 |
|---|---|
| Phổ biến toàn cục | 0,135 |
| FP-Growth, đặt luật danh mục trước | 0,227 |
| Phổ biến cùng danh mục | 0,247 |
| **FP-Growth, đặt "phổ biến cùng danh mục" trước** (`min_count = 2`) | **0,308** |

- Thứ tự fallback rất quan trọng: đổi thứ tự làm HitRate@10 tụt từ 0,308 xuống 0,227.
- *Đang chờ:* bản dùng lịch sử tháng 10+11. Trên validation, bản này tốt hơn rõ (0,270 so với 0,256, số luật 869 so với 480). Kết quả test sẽ có khi thí nghiệm chạy xong.

## 5. Phát hiện đáng đưa vào báo cáo

1. **Khách hay quay lại xem sản phẩm cũ.** 14% sản phẩm tiếp theo là sản phẩm đã xem trước đó trong phiên. Chỉ riêng việc đặt chúng lên đầu đã tăng HitRate@10 thêm khoảng 4 điểm. Bộ xếp hạng còn hơn quy tắc đó thêm 2,95 điểm [+2,74; +3,12]. Đặc trưng quan trọng nhất của nó là `steps_back` (số bước kể từ lần xem trước).
2. **Ý tưởng trong `note_hgy/Note.md` có tác dụng:** chèn token danh mục và thương hiệu vào chuỗi huấn luyện tăng HitRate@10 của Item2Vec thêm 35%. Dù vậy, Item2Vec vẫn thua co-visitation. Đã thử `window` 2, `ns_exponent` −0,5, 20 epoch, `sample` 1e-4 và 1e-5; tất cả đều kém cấu hình mặc định.
3. **Sản phẩm thay thế hay bổ sung:** 98% gợi ý của FP-Growth cùng danh mục với sản phẩm đang xem (sản phẩm thay thế), trong khi ở giỏ hàng thật tỉ lệ này là 67%. Metric thưởng cho sản phẩm thay thế, nhưng mục tiêu kinh doanh của "thường được mua cùng" là sản phẩm bổ sung (ốp lưng, tai nghe).
4. **Dữ liệu tháng 10 có tác dụng khác nhau tuỳ mô-đun:**
   - Giúp FP-Growth (thêm luật) và giảm tỉ lệ người dùng chưa có lịch sử từ 34% xuống 25%.
   - Không giúp co-visitation.
   - Làm baseline theo danh mục kém đi, vì sở thích một tháng trước kém chính xác hơn. ALS nên giảm trọng số theo thời gian.
5. **Đuôi dài rất nặng:** 1% sản phẩm hot nhất chiếm 41% tương tác. Baseline phổ biến có độ phủ catalog chỉ 0,01%, trong khi bộ gợi ý theo phiên phủ 40–47%.
6. **Giới hạn của đánh giá offline (slide 29):** mọi sản phẩm chưa tương tác bị coi là "không phù hợp", nên điểm tuyệt đối thấp hơn thực tế. Đánh giá offline chỉ phù hợp để so sánh các mô hình với nhau.

## 6. Demo Streamlit (`app.py`)

- **Trang chủ theo 3 loại người dùng:**
  - Khách vãng lai: gợi ý theo phiên, hoặc "Đang thịnh hành" khi phiên rỗng.
  - Thành viên mới: mô hình tạm (phổ biến theo danh mục yêu thích).
  - Thành viên cũ: mô hình tạm, chờ ALS.
- **Trang sản phẩm:** "Sản phẩm liên quan" (Item2Vec) và "Thường được mua cùng" (FP-Growth).
- **Giỏ hàng:** gợi ý mua kèm.
- Mỗi gợi ý kèm lý do. Sản phẩm có luật mua kèm được đánh dấu ★.
- Đã kiểm tra mọi luồng bằng `streamlit.testing.AppTest`.

## 7. Sự cố và cách xử lý trong ngày

| Vấn đề | Xử lý |
|---|---|
| Máy chưa có `pip`/`venv`, không chạy được `sudo` qua Claude | Bạn tự cài `python3-venv`, `python3-pip` trong terminal |
| gensim mặc định bỏ bớt token quá thường xuyên (`sample=1e-3`), làm test trên dữ liệu nhỏ thất bại | Đưa `sample` thành tham số của `Item2Vec` |
| Lần đầu dò tham số trên chính tập test | Tách tuần validation; test chỉ chạy một lần |
| Chọn dữ liệu (tháng 11 hay 10+11) bằng một tiêu chí chung cho mọi mô-đun | Mỗi mô-đun chọn theo metric của chính nó trên validation |
| Thí nghiệm bị ngắt khi phiên làm việc đóng | Đang chạy lại |

## 8. Việc còn lại

- [ ] Lấy kết quả test của FP-Growth với lịch sử tháng 10+11 (thí nghiệm đang chạy).
- [ ] Build lại demo có bộ xếp hạng LTR.
- [ ] Phối hợp với Dương:
  - gắn ALS và content-based vào khung đánh giá (chỉ cần `fit(train)` và `recommend(user_ids, k)`);
  - thay hai mô hình tạm trong demo;
  - đưa ALS vào bộ xếp hạng như một nguồn ứng viên.
- [ ] (Tuỳ chọn) Tách gợi ý sản phẩm thay thế và sản phẩm bổ sung cho "thường được mua cùng".
- [ ] (Tuỳ chọn) SASRec cho gợi ý theo phiên.

## 9. Cách chạy lại

```bash
.venv/bin/python src/make_sample.py archive.zip 2019-Nov.csv data/nov_u5.parquet --pct 5
.venv/bin/python src/make_sample.py archive.zip 2019-Oct.csv data/oct_u5.parquet --pct 5
.venv/bin/python src/experiments.py      # validation + test, ~25 phút → results/experiments.txt
.venv/bin/python src/ltr_session.py data/nov_u5.parquet   # bộ xếp hạng → results/ltr_session.txt
.venv/bin/python src/build_artifacts.py  # mô hình cho demo
.venv/bin/streamlit run app.py
.venv/bin/python -m pytest -q tests      # 12 test
```

**File chính:**

| File | Nội dung |
|---|---|
| `src/evalkit.py` | Chia dữ liệu, metric theo L8, bootstrap |
| `src/baselines.py` | Baseline phổ biến |
| `src/fbt.py` | FP-Growth |
| `src/item2vec.py` | Item2Vec, co-visitation, mô hình lai |
| `src/ltr_session.py` | Bộ xếp hạng LTR |
| `src/experiments.py` | Toàn bộ thí nghiệm validation + test |
| `app.py` | Demo Streamlit |
| `results/` | Log và kết quả |
