# Báo cáo tiến độ – Hệ gợi ý lai cho sàn TMĐT

*Cập nhật: 24/11/2026 · Dữ liệu: REES46 eCommerce behavior data (Kaggle), tháng 11/2019*

## 1. Tóm tắt

- **Dữ liệu và EDA:** xong. Dữ liệu rất thưa, chỉ ~10% người dùng có mua hàng, 44% giao dịch ở tuần test đến từ người dùng chưa có lịch sử. Từ 21/11 nhóm đã chuyển sang lấy mẫu 5% **theo người dùng** (của Khôi) thay cho lấy mẫu 10% theo dòng.
- **Các mô-đun đã có và đã đánh giá trên tuần test (24–30/11):**

  | Mô-đun | Mô hình | HitRate@10 | Baseline tốt nhất |
  |---|---|---|---|
  | Khách vãng lai | Bộ xếp hạng LightGBM (Khôi) | 0,345 | Co-visitation 0,262 |
  | Thường được mua cùng | FP-Growth + fallback (Khôi) | 0,308 | Phổ biến cùng danh mục 0,247 |
  | Thành viên cũ + mới (có lịch sử) | **Router: ALS + content-based (Dương)** | **0,259** (toàn bộ người dùng test) | `PopularByCategory` 0,187 |

- **Bộ định tuyến** chọn mô-đun theo nhóm người dùng, chọn trên tuần validation, test chạy một lần; hơn baseline +0,072 HitRate@10 và hơn ALS đơn lẻ +0,006 (cả hai có ý nghĩa thống kê, p < 0,001).
- **Đột biến 14–17/11 đã điều tra:** không phải bot, dữ liệu giữ nguyên (mục 3.4).
- **Còn lại:** nhóm người dùng chưa có lịch sử (34%) chỉ đạt mức baseline (không có thông tin nào để cá nhân hoá); báo cáo cuối kỳ và slide.

## 2. Đã làm

| Việc | Kết quả |
|---|---|
| Giải nén `archive.zip` | `raw_data/2019-Oct.csv` (5,7 GB), `raw_data/2019-Nov.csv` (9,0 GB) – đã kiểm tra đúng dung lượng |
| Notebook làm sạch + EDA (do thành viên viết notebook gốc soạn) | Chạy xong từ đầu đến cuối trên máy local |
| Chỉnh sửa notebook | Đường dẫn local thay cho `/kaggle/input`; đọc CSV theo chunk với kiểu dữ liệu tối ưu (máy chỉ có ~5 GB RAM); thêm bước lưu dữ liệu; thêm ô thống kê EDA cho bài toán gợi ý; sửa lỗi `fillna` trên cột categorical |
| Dữ liệu sạch | `cleaned_data/events_clean_all.parquet` – 6,73 triệu dòng, 1,83 triệu người dùng (dùng cho cold-start/guest) · `cleaned_data/events_filtered.parquet` – 4,19 triệu dòng, 394.880 người dùng (5–200 tương tác), 135.842 sản phẩm |

**Các bước làm sạch:** lấy mẫu 10% dòng của tháng 11 (6.752.985 / 67.501.979 dòng) → ép kiểu → bỏ 18.909 dòng giá ≤ 0 → tách `category_code` thành `main_category` / `sub_category` / `item_type` → điền `unknown` cho brand/category thiếu → gán điểm tương tác (view 1, cart 3, purchase 5) → sắp xếp theo user, thời gian → lọc người dùng có 5–200 tương tác.

**Từ 22–24/11:**
- Khôi: khung đánh giá theo slide L8, baseline, FP-Growth, Item2Vec + co-visitation + bộ xếp hạng LTR cho khách vãng lai, demo Streamlit (xem `BAO_CAO_TIEN_DO_KHOI.md`).
- Dương (demo): làm lại giao diện Streamlit sau khi hai chuyên gia (UI/UX và hệ gợi ý) review: thẻ sản phẩm, lý do gợi ý thật cho từng dòng, tách luật "mua cùng" khỏi phần bổ sung, giỏ hàng tương tác, tab Đánh giá (`src/make_summary.py` → `results/summary.json`).
- Dương: `src/als.py` + `src/run_als.py` (ALS), `src/content_based.py` (content-based), `src/router.py` + `src/run_router.py` (bộ định tuyến), ghép vào `src/build_artifacts.py` và `app.py`; kết quả ở `results/als.txt`, `results/router.txt`. 17/18 test qua.

> Lưu ý: dữ liệu thô và dữ liệu sạch **không nằm trong repo** (quá nặng). Cách tái tạo ở mục 7.

## 3. Phát hiện từ EDA

*Các con số dưới đây tính trên mẫu 10% theo dòng; riêng các số "toàn bộ người dùng" tính trên `events_clean_all`.*

### 3.1 Hành vi và độ thưa

| Phát hiện | Số liệu |
|---|---|
| Cơ cấu sự kiện | view 94,3% · cart 4,5% · purchase 1,3% (dữ liệu **không có** `remove_from_cart`) |
| Ma trận user × item | 394.880 × 135.842, 3,44 triệu cặp duy nhất, độ thưa **99,994%** |
| Cặp user–item có >1 sự kiện | chỉ 14,8% |
| Phân phối long-tail | top 1% sản phẩm chiếm 41,2% tương tác; top 20% chiếm 86,1%; **46,6% sản phẩm có <5 tương tác** |
| Người dùng mua hàng | chỉ **10,0%** (39.406 người, 52.825 lượt mua, ~1,2 lượt/người; 12,6% người mua có ≥2 lượt) |
| Mức độ hoạt động | 78,4% người dùng có <5 tương tác (nắm 37,5% số sự kiện) – nhóm này bị bộ lọc 5–200 loại bỏ |

### 3.2 Chất lượng thông tin sản phẩm

| Trường | Tỉ lệ `unknown` |
|---|---|
| `item_type` | **70,7%** (74,6% giao dịch mua thiếu trường này) |
| `main_category` | 32,8% |
| `brand` | 14,1% |
| `price`, `category_id` | không thiếu |

→ Nội dung sản phẩm cho mô-đun content-based nên dựa vào **`category_id`, `brand`, khoảng giá**, không dựa vào `category_code`.

### 3.3 Giỏ hàng ("thường được mua cùng")

- Chỉ **1.081 / 51.077** phiên có mua (2,1%) chứa ≥2 sản phẩm khác nhau → luật FP-Growth trên mẫu hiện tại sẽ rất nghèo.
- Nguyên nhân chính là lấy mẫu **theo dòng**: mỗi giỏ hàng bị cắt rụng, số cặp đồng xuất hiện giảm mạnh.

### 3.4 Thời gian và phiên

- **Đột biến lưu lượng 14–17/11 (đã điều tra, trên mẫu 5%):** 306 nghìn sự kiện toàn bộ dữ liệu ngày 14/11 và 618–646 nghìn ngày 15–17/11, gấp ~3,6 lần trung vị (~174 nghìn/ngày).
  - Không phải bot hay vài người dùng bất thường: số người dùng/ngày tăng 1,5 lần (14,5k → 22,5k), tỉ lệ người dùng top 1% và sản phẩm top 1% gần như không đổi; 73% người dùng của đợt này cũng hoạt động ngoài 14–17/11.
  - Có dấu hiệu khuyến mãi: lượt thêm giỏ gấp ~6 lần ngày 15/11, tỉ lệ cart+purchase 9,5% (so với 5,1%).
  - **Sự kiện `purchase` ngày 15/11 bằng 0**, rồi tăng vọt 16–17/11 (gấp 3 đến 8 lần bình thường) → nghi khoảng trống ghi log; chưa xác định được nguyên nhân.
  - **Quyết định: giữ nguyên dữ liệu.** Thử bỏ 14–17/11 khỏi huấn luyện ALS làm NDCG@10 validation giảm từ 0,161 xuống 0,120; bỏ riêng ngày 15 giảm xuống 0,159.
- Phiên: trong các phiên có >1 sự kiện, chỉ 0,1% kéo dài hơn 1 ngày, 8,7% kéo dài hơn 30 phút → `user_session` nhìn chung dùng được trực tiếp cho mô hình theo phiên (không cần cắt lại phiên).
- Top thương hiệu: samsung, apple, xiaomi, huawei, lucente, lg, bosch, respect, lenovo, sony. Ngành hàng lớn nhất: electronics (32,8%), appliances (13,5%), computers (7,0%).

### 3.5 Khả năng đánh giá theo thời gian

Nếu chia train đến 23/11 và test 24–30/11: có **70.812** sự kiện cart/purchase ở tuần test, trong đó **99,1% rơi vào sản phẩm đã xuất hiện ở train** nhưng **chỉ 55,7% đến từ người dùng đã có lịch sử**. Nghĩa là ALS chỉ phục vụ được khoảng một nửa số chuyển đổi; 44% còn lại cần mô-đun cold-start/guest. Phải báo cáo kết quả **theo từng nhóm người dùng**, không chỉ trung bình.

## 4. Hệ quả cho thiết kế

1. ✅ Đã đổi sang **lấy mẫu theo `user_id`** (hash), hoặc chạy FP-Growth trên toàn bộ dữ liệu theo chunk.
2. Giữ `events_clean_all` cho mô-đun guest / thành viên mới; chỉ dùng `events_filtered` cho ALS.
3. Dùng **cart + purchase làm nhãn dương** khi đánh giá (purchase quá ít).
4. Với ALS: giới hạn/log-scale số lần view lặp lại, giảm trọng số theo thời gian, bỏ sản phẩm <5 tương tác.
5. ✅ Đột biến 14–17/11: giữ nguyên (mục 3.4).
6. ✅ Dữ liệu tháng 10 làm lịch sử: giúp FP-Growth (thêm luật) nhưng làm ALS kém đi (NDCG@10 validation 0,111 so với 0,125) → ALS chỉ dùng tháng 11.

## 5. Tiến độ công việc

**Giai đoạn 1 – Dữ liệu**
- [x] Lấy mẫu theo người dùng (5%), xuất lại parquet (Khôi, `src/make_sample.py`)
- [x] Tìm hiểu đột biến 14–17/11 và quyết định xử lý (giữ nguyên)
- [x] Kiểm tra mức trùng người dùng giữa tháng 10 và 11 (Khôi: tỉ lệ người dùng tuần test đã có lịch sử tăng 66,4% → 73,6%)

**Giai đoạn 2 – Đánh giá**
- [x] Khung đánh giá, chia thời gian, metric theo L8, bootstrap (Khôi)
- [x] Baseline phổ biến 7 ngày và theo danh mục (Khôi)

**Giai đoạn 3 – Mô-đun gợi ý**
- [x] Thành viên cũ: ALS – HitRate@10 0,285 trên người dùng có lịch sử (baseline 0,186)
- [x] Thành viên mới: content-based (`category_id`, `brand`, khoảng giá)
- [x] Khách vãng lai: co-visitation, Item2Vec, bộ xếp hạng LTR (Khôi)
- [x] "Thường được mua cùng": FP-Growth (Khôi)
- [x] Bộ định tuyến + trộn điểm theo thứ hạng (RRF), chọn trên validation
- [ ] Gắn ALS vào bộ xếp hạng LTR của khách vãng lai như một nguồn ứng viên (Khôi đề xuất, tuỳ chọn)
- [ ] Tách gợi ý sản phẩm thay thế / bổ sung cho "thường được mua cùng" (tuỳ chọn)

**Giai đoạn 4 – Hoàn thiện**
- [x] Demo Streamlit có đủ 3 loại người dùng, trang sản phẩm, giỏ hàng
- [ ] Báo cáo cuối kỳ và slide

## 6. Kết quả của bộ định tuyến (Dương)

Test 24–30/11, 10.101 người dùng có cart/purchase, mẫu 5% người dùng, mọi lựa chọn chốt trên validation 17–23/11:

| Nhóm người dùng | Mô-đun được chọn | HitRate@10 | NDCG@10 |
|---|---|---|---|
| Chưa có lịch sử (`unseen`, 34%) | PopularRecent | 0,188 | 0,095 |
| 1–4 tương tác (`light`, 11%) | Fusion(ALS×2, Content×1) | 0,226 | 0,143 |
| ≥ 5 tương tác (`heavy`, 55%) | Fusion(ALS×2, Content×1) | 0,310 | 0,172 |
| **Toàn bộ** | **Router** | **0,259** [0,251; 0,268] | **0,143** |

So sánh: `PopularByCategory` 0,187 · ALS đơn lẻ 0,253 · content-based đơn lẻ 0,233 (HitRate@10 toàn bộ).

Lưu ý khi đọc kết quả:
- Mô hình thu được một phần là gợi ý "quay lại sản phẩm đã xem" (không loại sản phẩm đã tương tác được chọn vì nhãn gồm cả các lần lặp lại). 90% gợi ý nằm trong 1% sản phẩm hot nhất, độ phủ catalog 2,6%.
- Nhóm `unseen` không có thông tin nào về người dùng nên chỉ có thể dùng độ phổ biến; trong sản phẩm thật, danh mục yêu thích chọn lúc đăng ký (demo đã hỗ trợ) là tín hiệu duy nhất.
- Đánh giá offline coi mọi sản phẩm chưa tương tác là "không phù hợp", nên điểm tuyệt đối thấp hơn thực tế; chỉ dùng để so sánh các mô hình với nhau.

## 7. Cách chạy lại / lưu ý kỹ thuật

- Dữ liệu thô tải từ Kaggle (`mkechinov/ecommerce-behavior-data-from-multi-category-store`), giải nén vào `raw_data/`. `raw_data/` và `cleaned_data/` nằm ngoài repo, chưa được đưa lên Git.
- Đường dẫn trong ô đầu của notebook (`RAW_PATH`, `CLEAN_DIR`) đang trỏ tới `/home/duong/RS/...` – mỗi người cần sửa lại theo máy mình.
- Notebook cần: `pandas`, `numpy`, `pyarrow`, `plotly`; để chạy bằng dòng lệnh cần thêm `jupyter`/`nbconvert`. Các mô-đun mô hình cần thêm `implicit`, `gensim`, `mlxtend`, `lightgbm`, `streamlit`, `pytest`.
- Máy hiện tại có ~5 GB RAM nên notebook đọc file theo chunk; không nên đọc nguyên file CSV bằng `pd.read_csv` không kèm `chunksize`.
- Thư mục `notebook/iframe_figures/` (~44 MB) do plotly tạo khi chạy notebook; đã thêm vào `.gitignore`, biểu đồ sẽ không hiển thị khi xem notebook trên GitHub trừ khi chạy lại.

**Chạy lại phần mô hình** (mẫu 5% theo người dùng; `OPENBLAS_NUM_THREADS=1` để `implicit` không tranh luồng BLAS):

```bash
.venv/bin/python src/make_sample.py archive.zip 2019-Nov.csv data/nov_u5.parquet --pct 5   # hoặc đường dẫn tới file .csv đã giải nén
.venv/bin/python src/make_sample.py archive.zip 2019-Oct.csv data/oct_u5.parquet --pct 5
export PYTHONPATH=src OPENBLAS_NUM_THREADS=1
.venv/bin/python src/run_als.py        # chọn tham số ALS trên validation, test một lần → results/als.txt (~5 phút)
.venv/bin/python src/run_router.py     # content-based + chọn mô-đun theo nhóm + test → results/router.txt (~3 phút)
.venv/bin/python src/build_artifacts.py && .venv/bin/streamlit run app.py
.venv/bin/python -m pytest -q tests    # 18 test
```
