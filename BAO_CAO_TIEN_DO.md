# Báo cáo tiến độ – Hệ gợi ý lai cho sàn TMĐT

*Cập nhật: 21/11/2026 · Dữ liệu: REES46 eCommerce behavior data (Kaggle), tháng 11/2019*

## 1. Tóm tắt

- Đã giải nén dữ liệu thô (tháng 10 và 11/2019), chạy xong notebook làm sạch + EDA, và lưu dữ liệu sạch ra file parquet.
- EDA cho thấy dữ liệu **rất thưa**, **chỉ ~10% người dùng có mua hàng**, **44% giao dịch ở tuần test đến từ người dùng chưa có lịch sử**, và thông tin danh mục/thương hiệu bị thiếu nhiều. Điều này củng cố hướng đi lai (hybrid) có định tuyến theo nhóm người dùng.
- Phát hiện quan trọng nhất: **cách lấy mẫu 10% theo dòng đang làm hỏng dữ liệu giỏ hàng** (chỉ còn 1.081 phiên mua có ≥2 sản phẩm). Cần đổi sang lấy mẫu theo người dùng trước khi huấn luyện mô hình.
- Chưa có mô hình nào được huấn luyện. Phần còn lại và đề xuất phân công ở mục 5–6.

## 2. Đã làm

| Việc | Kết quả |
|---|---|
| Giải nén `archive.zip` | `raw_data/2019-Oct.csv` (5,7 GB), `raw_data/2019-Nov.csv` (9,0 GB) – đã kiểm tra đúng dung lượng |
| Notebook làm sạch + EDA (do thành viên viết notebook gốc soạn) | Chạy xong từ đầu đến cuối trên máy local |
| Chỉnh sửa notebook | Đường dẫn local thay cho `/kaggle/input`; đọc CSV theo chunk với kiểu dữ liệu tối ưu (máy chỉ có ~5 GB RAM); thêm bước lưu dữ liệu; thêm ô thống kê EDA cho bài toán gợi ý; sửa lỗi `fillna` trên cột categorical |
| Dữ liệu sạch | `cleaned_data/events_clean_all.parquet` – 6,73 triệu dòng, 1,83 triệu người dùng (dùng cho cold-start/guest) · `cleaned_data/events_filtered.parquet` – 4,19 triệu dòng, 394.880 người dùng (5–200 tương tác), 135.842 sản phẩm |

**Các bước làm sạch:** lấy mẫu 10% dòng của tháng 11 (6.752.985 / 67.501.979 dòng) → ép kiểu → bỏ 18.909 dòng giá ≤ 0 → tách `category_code` thành `main_category` / `sub_category` / `item_type` → điền `unknown` cho brand/category thiếu → gán điểm tương tác (view 1, cart 3, purchase 5) → sắp xếp theo user, thời gian → lọc người dùng có 5–200 tương tác.

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

- **Đột biến lưu lượng 14–17/11:** 306 nghìn (14/11), rồi 618–646 nghìn sự kiện/ngày (15–17/11), trong khi trung vị cả tháng là ~174 nghìn/ngày (gấp ~3,6 lần). Chưa rõ nguyên nhân (khuyến mãi hay traffic bất thường) – cần kiểm tra trước khi huấn luyện.
- Phiên: trong các phiên có >1 sự kiện, chỉ 0,1% kéo dài hơn 1 ngày, 8,7% kéo dài hơn 30 phút → `user_session` nhìn chung dùng được trực tiếp cho mô hình theo phiên.
- Top thương hiệu: samsung, apple, xiaomi, huawei, lucente, lg, bosch, respect, lenovo, sony. Ngành hàng lớn nhất: electronics (32,8%), appliances (13,5%), computers (7,0%).

### 3.5 Khả năng đánh giá theo thời gian

Nếu chia train đến 23/11 và test 24–30/11: có **70.812** sự kiện cart/purchase ở tuần test, trong đó **99,1% rơi vào sản phẩm đã xuất hiện ở train** nhưng **chỉ 55,7% đến từ người dùng đã có lịch sử**. Nghĩa là ALS chỉ phục vụ được khoảng một nửa số chuyển đổi; 44% còn lại cần mô-đun cold-start/guest. Phải báo cáo kết quả **theo từng nhóm người dùng**, không chỉ trung bình.

## 4. Hệ quả cho thiết kế

1. Đổi sang **lấy mẫu theo `user_id`** (hash), hoặc chạy FP-Growth trên toàn bộ dữ liệu theo chunk.
2. Giữ `events_clean_all` cho mô-đun guest / thành viên mới; chỉ dùng `events_filtered` cho ALS.
3. Dùng **cart + purchase làm nhãn dương** khi đánh giá (purchase quá ít).
4. Với ALS: giới hạn/log-scale số lần view lặp lại, giảm trọng số theo thời gian, bỏ sản phẩm <5 tương tác.
5. Cân nhắc bỏ hoặc giảm trọng số 14–17/11.
6. Có thể dùng dữ liệu tháng 10 làm lịch sử huấn luyện và tháng 11 làm test (chưa kiểm tra tỉ lệ trùng người dùng giữa hai tháng).

## 5. Công việc còn lại

**Giai đoạn 1 – Dữ liệu**
- [ ] Lấy mẫu lại theo người dùng, chạy lại notebook, xuất lại parquet
- [ ] Tìm hiểu nguyên nhân đột biến 14–17/11 và quyết định cách xử lý
- [ ] (Tùy chọn) Kiểm tra mức trùng người dùng giữa tháng 10 và 11

**Giai đoạn 2 – Đánh giá**
- [ ] Khung đánh giá: chia train/test theo thời gian, Recall/NDCG/HitRate/MRR@K (K=10, 20), báo cáo theo nhóm người dùng
- [ ] Baseline: phổ biến nhất 7 ngày gần nhất; phổ biến theo ngành hàng

**Giai đoạn 3 – Mô-đun gợi ý**
- [ ] Thành viên cũ: ALS (implicit)
- [ ] Thành viên mới: content-based (category_id, brand, khoảng giá) + bán chạy theo danh mục
- [ ] Khách vãng lai: co-visitation / Item2Vec (gensim); SR-GNN là phần mở rộng, tùy chọn
- [ ] "Thường được mua cùng": FP-Growth (mlxtend), báo cáo support/confidence/lift
- [ ] Bộ định tuyến theo số tương tác + trộn điểm; tinh chỉnh trọng số trên tuần validation

**Giai đoạn 4 – Hoàn thiện**
- [ ] Demo Streamlit (có thể làm sớm với dữ liệu giả)
- [ ] Báo cáo kết quả và slide

## 6. Đề xuất phân công

Nguyên tắc: phần việc của Khôi **độc lập**, có đầu vào/đầu ra rõ, không chặn công việc của người khác; phần việc của Dương là phần lõi và những chỗ cần hiểu rõ dữ liệu.

| Người | Việc | Ghi chú |
|---|---|---|
| **Khôi** | 1. Khung đánh giá + baseline phổ biến | Làm đầu tiên, cả nhóm đều cần. Chạy được trên bất kỳ file parquet cùng schema nên không phải chờ lấy mẫu lại |
| | 2. Mô-đun FP-Growth "thường được mua cùng" | Độc lập; cần dữ liệu đã lấy mẫu lại theo user hoặc toàn bộ dữ liệu |
| | 3. Mô-đun Item2Vec cho khách vãng lai | Đầu vào là chuỗi sản phẩm theo `user_session` |
| | 4. Demo Streamlit | Bắt đầu với dữ liệu giả, ghép mô hình thật sau |
| **Dương** | 1. Lấy mẫu lại theo user, chạy lại notebook, xuất lại parquet | Làm trước để Khôi dùng dữ liệu mới cho FP-Growth |
| | 2. Kiểm tra đột biến 14–17/11 | |
| | 3. ALS cho thành viên cũ | |
| | 4. Content-based cho thành viên mới | |
| | 5. Bộ định tuyến + trộn điểm | Cần mô-đun của Khôi làm xong |

Nếu thành viên viết notebook gốc còn thời gian, có thể nhận lại mô-đun content-based hoặc Item2Vec.

**Thứ tự đề xuất:** Dương lấy mẫu lại dữ liệu và Khôi dựng khung đánh giá (song song) → Dương làm ALS, Khôi làm FP-Growth → content-based, Item2Vec → ghép bằng bộ định tuyến → demo và báo cáo.

## 7. Cách chạy lại / lưu ý kỹ thuật

- Dữ liệu thô tải từ Kaggle (`mkechinov/ecommerce-behavior-data-from-multi-category-store`), giải nén vào `raw_data/`. `raw_data/` và `cleaned_data/` nằm ngoài repo, chưa được đưa lên Git.
- Đường dẫn trong ô đầu của notebook (`RAW_PATH`, `CLEAN_DIR`) đang trỏ tới `/home/duong/RS/...` – mỗi người cần sửa lại theo máy mình.
- Notebook cần: `pandas`, `numpy`, `pyarrow`, `plotly`; để chạy bằng dòng lệnh cần thêm `jupyter`/`nbconvert`. Các mô-đun mô hình cần thêm `implicit`, `gensim`, `mlxtend`, `streamlit` (chưa cài).
- Máy hiện tại có ~5 GB RAM nên notebook đọc file theo chunk; không nên đọc nguyên file CSV bằng `pd.read_csv` không kèm `chunksize`.
- Thư mục `notebook/iframe_figures/` (~44 MB) do plotly tạo khi chạy notebook; đã thêm vào `.gitignore`, biểu đồ sẽ không hiển thị khi xem notebook trên GitHub trừ khi chạy lại.
