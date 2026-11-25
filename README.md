# Hệ gợi ý lai cho sàn TMĐT – phần việc của Khôi

Gồm 4 việc trong mục 6 của `BAO_CAO_TIEN_DO.md`: khung đánh giá + baseline, FP-Growth
"thường được mua cùng", Item2Vec cho khách vãng lai, demo Streamlit.

## Cài đặt và chạy

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt   # pandas, implicit, lightgbm, gensim, mlxtend, streamlit, ...

# 1. Lấy mẫu 5% người dùng tháng 11, đọc thẳng từ zip (không cần giải nén 14,7 GB)
.venv/bin/python src/make_sample.py archive.zip 2019-Nov.csv data/nov_u5.parquet --pct 5

# 2. Đánh giá (train đến 23/11, test 24–30/11)
.venv/bin/python src/run_baselines.py data/nov_u5.parquet
.venv/bin/python src/run_fbt.py       data/nov_u5.parquet
.venv/bin/python src/run_item2vec.py  data/nov_u5.parquet

# 3. Demo (xem mục "Demo Streamlit" bên dưới)
OPENBLAS_NUM_THREADS=1 .venv/bin/python src/build_artifacts.py --nov data/nov_u5.parquet --octnov data/octnov_u5.parquet
.venv/bin/streamlit run app.py

# Test
.venv/bin/python -m pytest -q tests
```

Mọi script nhận bất kỳ parquet nào có cùng schema REES46, nên chạy được ngay trên
dữ liệu Dương xuất lại.

## Cấu trúc

| File | Nội dung |
|---|---|
| `src/make_sample.py` | Lấy mẫu theo hash `user_id` (giữ nguyên lịch sử và giỏ hàng của người được chọn) |
| `src/evalkit.py` | Chia train/test theo thời gian, Recall/NDCG/HitRate/MRR@K, báo cáo theo nhóm `unseen` / `light` (1–4 tương tác) / `heavy` (≥5); nhãn dương = cart + purchase |
| `src/baselines.py` | `PopularRecent` (phổ biến 7 ngày), `PopularByCategory` (phổ biến trong `category_id` yêu thích) |
| `src/fbt.py` | FP-Growth (mlxtend): luật sản phẩm → phổ biến cùng danh mục → luật danh mục → thịnh hành |
| `src/item2vec.py` | `Item2Vec` (có `side_info`: chèn token danh mục/thương hiệu), `CoVisitation`, `CoVisitThenItem2Vec` |
| `src/build_artifacts.py`, `app.py` | Huấn luyện + lưu mô hình, demo Streamlit |

## Kết quả (mẫu 5% người dùng tháng 11)

**Gợi ý Top-K cho người dùng** (10.101 người có cart/purchase ở tuần test):

| Mô hình | HitRate@10 | NDCG@10 | Recall@20 |
|---|---|---|---|
| PopularRecent | 0,172 | 0,080 | 0,186 |
| PopularByCategory | 0,187 | 0,093 | 0,200 |

Mô hình của nhóm (ALS, content-based, router) phải vượt `PopularByCategory`, và nên
báo cáo theo từng nhóm người dùng (`unseen` chiếm 34% số người dùng test).

**Thường được mua cùng** (giỏ = cart + purchase cùng phiên; 10.530 giỏ train, 451 luật
sản phẩm, 288 luật danh mục; đánh giá leave-one-out trên 1.833 giỏ test):

| Hệ thống | HitRate@5 | HitRate@10 | MRR@10 |
|---|---|---|---|
| **FP-Growth + fallback** | **0,222** | **0,288** | **0,150** |
| Phổ biến cùng danh mục | 0,174 | 0,247 | 0,111 |
| Phổ biến toàn cục | 0,090 | 0,135 | 0,059 |

Sản phẩm có luật (36% truy vấn) đạt HitRate@10 = 0,455. Thứ tự fallback quan trọng:
đặt luật danh mục trước "phổ biến cùng danh mục" làm HitRate@10 tụt còn 0,187.

**Gợi ý theo phiên (khách vãng lai)**, đoán sản phẩm tiếp theo, 20.000 phiên test
(89.460 truy vấn):

| Hệ thống | HitRate@10 | MRR@10 | HitRate@20 |
|---|---|---|---|
| Phổ biến | 0,035 | 0,012 | 0,055 |
| Item2Vec | 0,157 | 0,074 | 0,215 |
| Item2Vec + side_info | 0,210 | 0,101 | 0,277 |
| Co-visitation | 0,259 | 0,103 | 0,361 |
| **Co-visitation → Item2Vec** | **0,259** | **0,103** | **0,376** |

- `side_info` (ý trong `note_hgy/Note.md`) tăng HitRate@10 của Item2Vec thêm 35%.
- Đã thử `window` 2, `ns_exponent` −0,5, 20 epoch, `sample` 1e-4/1e-5: đều kém cấu hình
  mặc định (`window=5`, `ns_exponent=0.75`, `sample=1e-3`).
- Co-visitation thắng ở bài toán đoán sản phẩm tiếp theo, nên mô-đun khách dùng bản lai;
  Item2Vec giữ 2 vị trí và dùng cho "Sản phẩm liên quan" ở trang sản phẩm.

## Ghép mô-đun của Dương

Khung đánh giá chỉ cần một object có:

```python
class MyModel:
    def fit(self, train: pd.DataFrame) -> None: ...
    def recommend(self, user_ids: list[int], k: int) -> dict[int, list[int]]: ...

from evalkit import evaluate, load_events, temporal_split
train, test = temporal_split(load_events("data/nov_u5.parquet"), "2019-11-24")
print(evaluate(MyModel(), train, test))   # bảng theo nhóm unseen / light / heavy
```

Thêm mô hình vào dict `models` trong `src/build_artifacts.py` rồi build lại artifacts; demo đọc mô hình
từ `artifacts/models.pkl`.

## Lưu ý

- Chưa xử lý đột biến 14–17/11 (việc của Dương); 7 ngày dùng cho baseline phổ biến
  (17–23/11) có chứa ngày 17/11.
- Khung đánh giá chưa có tùy chọn loại sản phẩm đã tương tác ở train; có thể cần cho ALS.
- `data/`, `artifacts/` (~430 MB), `archive.zip` không đưa lên Git.

## ALS cho thành viên cũ (Dương)

```bash
.venv/bin/pip install implicit
PYTHONPATH=src .venv/bin/python src/run_als.py     # chọn tham số trên validation, test một lần → results/als.txt
```

`src/als.py` có `ALSRecommender` đúng giao diện `fit(train)` / `recommend(user_ids, k)`; người dùng
chưa có lịch sử rơi về `PopularRecent`. Kết quả và các lựa chọn: xem `results/als.txt`.

## Content-based và bộ định tuyến (Dương)

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv/bin/python src/run_router.py   # → results/router.txt, results/router.json
```

- `src/content_based.py`: `ContentBased` – vector sản phẩm (`category_id`, `brand`, khoảng giá), hồ sơ người dùng từ lịch sử,
  cosine + độ phổ biến; `recommend_from_items` cho thành viên mới chọn danh mục/sản phẩm lúc đăng ký.
- `src/router.py`: `HybridRouter` (chọn mô-đun theo nhóm `unseen`/`light`/`heavy`), `RankFusion` (hợp nhất theo thứ hạng).
- `src/build_artifacts.py` đọc `results/router.json` (ALS, content-based, lựa chọn mô-đun) và demo `app.py` dùng router.

## Demo Streamlit

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python src/build_artifacts.py --nov data/nov_u5.parquet --octnov data/octnov_u5.parquet
                                            # huấn luyện mọi mô hình → artifacts/models.pkl (~430 MB, ~5 phút, không đưa lên Git)
.venv/bin/python src/make_summary.py        # gom số liệu test cho trang "Đánh giá" → results/summary.json
.venv/bin/streamlit run app.py              # http://localhost:8501
```

Hai trang (thanh điều hướng trên cùng):

**🛍️ Cửa hàng** – mô phỏng một lượt mua sắm, chọn vai trò ở đầu trang:
- *Khách vãng lai*: bấm **Xem** trên bất kỳ thẻ nào → mở hộp xem nhanh (sản phẩm tương tự Item2Vec, mua cùng FP-Growth,
  thêm vào giỏ) và ghi lượt xem vào **phiên** (dải chip phía trên, bỏ từng lượt được). Mọi kệ gợi ý xếp lại theo phiên
  bằng mô hình theo phiên (mặc định LightGBM re-ranker). "Tìm sản phẩm" để xem một mã cụ thể.
- **Đăng ký thành viên với phiên này**: khách thành *thành viên mới*, hồ sơ content-based dựng từ các sản phẩm vừa xem
  (cold-start hand-off). Không có phiên thì dùng danh mục yêu thích, mỗi danh mục xếp riêng rồi xen kẽ.
- *Thành viên cũ*: chọn tài khoản (nhiều / ít / chưa có lịch sử); bộ định tuyến chọn mô-đun theo nhóm, mỗi thẻ có nhãn
  thứ hạng trong ALS và content-based; nếu đang có phiên thì thêm kệ "Tiếp tục phiên".
- Khung **"Hệ thống đã làm gì"**: nhóm người dùng → luật định tuyến → mô-đun → nguồn ứng viên, kèm thời gian chạy
  (tắt được ở thanh bên).
- Cột **giỏ hàng** bên phải với gợi ý "Mua kèm" từ luật FP-Growth.

**📊 Đánh giá** – số liệu test từ `results/summary.json`: mỗi phần của hệ lai so với baseline, theo nhóm người dùng,
có khoảng tin cậy 95%.

Ghi chú:
- Sản phẩm, giá, thương hiệu, danh mục, lịch sử người dùng đều là dữ liệu REES46 thật; dữ liệu không có tên/ảnh nên
  tên hiển thị = thương hiệu + danh mục, biểu tượng và màu ô theo danh mục.
- Mỗi gợi ý kèm **bằng chứng thật** (thuộc tính trùng, "Đã xem / Mới", "% khách mua cùng", thứ hạng trong mô hình thành phần)
  thay cho điểm mô hình, vốn không so sánh được giữa các mô-đun.
- Thanh bên (mở bằng » góc trên trái): số gợi ý K, mô hình theo phiên, bật/tắt khung giải thích, "Giới hạn của demo".
- Giao diện: `.streamlit/config.toml` (màu, phông Be Vietnam Pro tải từ Google Fonts; không có mạng thì dùng phông mặc định).
- `tests/test_app.py` chạy thử các luồng chính của demo (bỏ qua nếu chưa build artifacts).
