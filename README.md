# Hệ gợi ý lai cho sàn TMĐT – phần việc của Khôi

Gồm 4 việc trong mục 6 của `BAO_CAO_TIEN_DO.md`: khung đánh giá + baseline, FP-Growth
"thường được mua cùng", Item2Vec cho khách vãng lai, demo Streamlit.

## Cài đặt và chạy

```bash
python3 -m venv .venv
.venv/bin/pip install pandas numpy pyarrow scipy mlxtend gensim streamlit pytest

# 1. Lấy mẫu 5% người dùng tháng 11, đọc thẳng từ zip (không cần giải nén 14,7 GB)
.venv/bin/python src/make_sample.py archive.zip 2019-Nov.csv data/nov_u5.parquet --pct 5

# 2. Đánh giá (train đến 23/11, test 24–30/11)
.venv/bin/python src/run_baselines.py data/nov_u5.parquet
.venv/bin/python src/run_fbt.py       data/nov_u5.parquet
.venv/bin/python src/run_item2vec.py  data/nov_u5.parquet

# 3. Demo
.venv/bin/python src/build_artifacts.py data/nov_u5.parquet artifacts/
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

Trong demo (`app.py`), hai chỗ đang dùng bản tạm, có ghi nhãn trên giao diện:
- `old_member_recs`: thay `A["old_member"]` (đang là `PopularByCategory`) bằng ALS.
- `new_member_recs`: thay bằng mô-đun content-based.

Thêm mô hình vào dict `models` trong `src/build_artifacts.py` rồi build lại artifacts.

## Lưu ý

- Chưa xử lý đột biến 14–17/11 (việc của Dương); 7 ngày dùng cho baseline phổ biến
  (17–23/11) có chứa ngày 17/11.
- Khung đánh giá chưa có tùy chọn loại sản phẩm đã tương tác ở train; có thể cần cho ALS.
- `data/`, `artifacts/` (~186 MB), `archive.zip` không đưa lên Git.
