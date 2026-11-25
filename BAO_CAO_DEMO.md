# Báo cáo: kiểm thử, sửa lỗi và thiết kế lại demo Streamlit

Nhánh: `duong-als-fixes` (dựa trên `duong-als`). Phạm vi: demo `app.py` và một lỗi trong `src/ltr_session.py`.
**Không thay đổi mô hình, tham số hay số liệu đánh giá** trong `results/`.

## 1. Nhánh của Dương đã có gì

| Commit | Nội dung |
|---|---|
| `a549f11` | ALS cho thành viên cũ (`src/als.py`, `src/run_als.py`): trọng số sự kiện, suy giảm theo thời gian, giới hạn lượt xem lặp, fallback phổ biến |
| `7e3a914` | Content-based (`src/content_based.py`), bộ định tuyến + hợp nhất thứ hạng (`src/router.py`, `src/run_router.py`) |
| `d5e3472` | Thiết kế lại demo Streamlit, tab Đánh giá đọc `results/summary.json` |

Kết quả test (tuần 24–30/11, 10.101 người dùng có thêm giỏ/mua):

| Mô hình | HitRate@10 | NDCG@10 |
|---|---|---|
| **Router** | **0,259** | **0,143** |
| ALS | 0,253 | 0,137 |
| ContentBased | 0,233 | 0,125 |
| PopularByCategory | 0,187 | 0,093 |

- Router hơn PopularByCategory +0,072 HitRate@10 (CI 95%: +0,066 … +0,079); hơn ALS chỉ +0,006.
- Nhóm chưa có lịch sử (34% người dùng test) nhận đúng kết quả baseline vì router gửi họ sang "thịnh hành 7 ngày".
- Độ phủ catalog 2,6%; ≈90% gợi ý thuộc top 1% sản phẩm phổ biến.

## 2. Cách kiểm thử

- `pytest -q tests`: 17 passed, 1 skipped (test demo bị bỏ qua vì chưa build artifacts) → build `artifacts/models.pkl` rồi chạy lại: 18 passed.
- Chạy demo trên Chrome, thử mọi vai trò, tab, giỏ hàng.
- Script quét mô hình: mọi lời gọi mô hình mà demo dùng, trên mẫu 1.500/12.795 sản phẩm, 200 phiên nhiều sản phẩm, 261 tài khoản demo
  (kiểm tra lỗi, danh sách rỗng, thiếu, trùng, gợi ý lại chính sản phẩm/giỏ).
- Script quét giao diện bằng `streamlit.testing.AppTest`: ~40 bước trên giao diện cũ, 127 trạng thái trên giao diện mới.
- Đo thời gian phản hồi trên trình duyệt và profile (`cProfile`) một lần chạy lại script.

## 3. Lỗi tìm thấy và cách sửa

### 3.1 Crash ở mô hình theo phiên mặc định (LightGBM re-ranker)
- **Hiện tượng**: phiên khách chỉ có sản phẩm `4201519` (hoặc `9001912`) → trang hiện traceback `IndexError`.
  9 sản phẩm demo bị ảnh hưởng (2 hiện mặc định, 7 khi bật "sản phẩm chưa phân loại").
- **Nguyên nhân**: sản phẩm không có hàng xóm co-visitation và không có vector Item2Vec → `build()` tạo bảng ứng viên rỗng,
  cột `qid` thành kiểu `object` và phép index numpy ở `src/ltr_session.py` lỗi trước khi tới nhánh fallback có sẵn.
- **Sửa**: tạo `qid` với dtype `int64`; fallback "thịnh hành" chạy đúng. Thêm test hồi quy trong `tests/test_item2vec.py`
  (test thất bại với code cũ, đạt với code mới).

### 3.2 Mỗi lần bấm mất ~2,8 s
- **Nguyên nhân**: `meta()` dùng `@lru_cache`, nhưng Streamlit chạy lại toàn bộ `app.py` sau mỗi tương tác nên hàm được định nghĩa
  lại với cache rỗng → ~23.000 lần tra `catalog.loc` mỗi lần bấm (3,4 s trên 4,2 s khi profile). Mô hình chỉ tốn < 0,05 s.
- **Sửa**: cache trong `st.cache_resource`; danh sách chọn sản phẩm, `HAS_RULE`, `POP_PCT` và các lời gọi mô hình cũng được cache.

### 3.3 Thành viên mới chỉ nhận một danh mục
- **Hiện tượng**: chọn electronics + apparel chỉ ra điện thoại Samsung; 24/78 cặp danh mục chỉ ra một danh mục;
  chọn riêng "stationery" ra 0/10 sản phẩm stationery (sport, accessories, medicine chỉ 20%).
- **Nguyên nhân**: (1) một hồ sơ trung bình cho mọi danh mục để danh mục lớn chiếm hết chỗ; (2) điểm cộng phổ biến (`beta=1`)
  kéo điện thoại vào danh mục nhỏ; (3) content-based chỉ chấm điểm pool 5.000 sản phẩm phổ biến gần đây, pool này có
  0–14 sản phẩm của các danh mục nhỏ.
- **Sửa (ở tầng hiển thị, không đổi mô hình)**: mỗi danh mục xếp riêng, giữ sản phẩm đúng danh mục lên đầu, bù bằng sản phẩm bán chạy
  của danh mục khi pool thiếu, rồi xen kẽ giữa các danh mục.
- **Kết quả**: 13/13 danh mục cho 100% sản phẩm đúng danh mục; 0/78 cặp chỉ ra một danh mục.

### 3.4 Lỗi khác
- Toast "đã thêm vào giỏ" gọi trong callback khi đang ở hộp thoại xem nhanh làm **trắng trang** (Streamlit không hỗ trợ vẽ phần tử
  trong callback khi chạy lại một fragment) → callback chỉ ghi thông báo, toast hiện lúc vẽ trang.
- Chuyển vai trò giữ nguyên tab con đang mở; trục biểu đồ khó đọc; nhãn số trên biểu đồ không thấy ở theme tối → đã sửa.

## 4. Thiết kế lại giao diện

Mục tiêu: demo phản ánh đúng câu chuyện của hệ lai – mỗi loại người dùng được phục vụ bởi mô-đun phù hợp, và chuyển giao khi họ
có thêm dữ liệu.

| Trước | Sau |
|---|---|
| Phiên khách chọn từ ô 12.795 sản phẩm | Bấm **Xem** trên thẻ → hộp xem nhanh + ghi vào phiên (chip bỏ được); mọi kệ xếp lại ngay |
| Tab Trang sản phẩm / Giỏ hàng lặp lại ở mỗi vai trò | Hộp xem nhanh (Item2Vec + FP-Growth) và cột giỏ hàng cố định với "Mua kèm" |
| Định tuyến chỉ là một dòng nhãn | Khung **"Hệ thống đã làm gì"**: nhóm người dùng → luật định tuyến → mô-đun → nguồn ứng viên, thời gian chạy |
| Khách và thành viên mới tách rời | **Đăng ký thành viên với phiên này**: hồ sơ content-based dựng từ phiên (cold-start hand-off) |
| Thẻ thành viên cũ: Đã xem / Mới | Thêm thứ hạng của sản phẩm trong ALS và content-based (thành phần của phép hợp nhất) |
| Tab Đánh giá: bảng thô | Trang riêng: thẻ số liệu chính (mỗi phần so với baseline), biểu đồ theo nhóm, khoảng tin cậy 95% |

Chi tiết giao diện: lưới thẻ co giãn theo bề ngang, thẻ cao bằng nhau, luôn đủ K sản phẩm mới, nhãn bằng chứng giống nhau ở mọi thẻ
gộp vào chú thích, thanh bên thu gọn mặc định, theme + phông Be Vietnam Pro (`.streamlit/config.toml`), hoạt động ở theme sáng và tối.

**Hiệu năng trên trình duyệt**

| Thao tác | Ban đầu | Sau sửa cache | Sau thiết kế lại |
|---|---|---|---|
| Thêm vào giỏ | 2,8 s | 0,75 s | 0,29 s |
| Mở xem nhanh | – | – | 0,34 s |

## 5. Kết quả kiểm thử cuối

- `pytest -q tests`: **19 passed** (`tests/test_app.py` viết lại cho luồng mới và trang Đánh giá; test hồi quy cho lỗi 3.1).
- Quét giao diện: **127 trạng thái, 0 lỗi** (3 vai trò × K × 4 mô hình theo phiên × các phiên gây crash cũ, 13 danh mục,
  61 tài khoản, xem nhanh, tìm kiếm).
- Kiểm tra thủ công trên Chrome ở theme sáng và tối.

## 6. Hạn chế còn lại và việc nên làm tiếp

1. **Thêm chỉ số "chỉ sản phẩm mới"**: nhãn test gồm cả sản phẩm đã tương tác nên mô hình được thưởng khi gợi ý lại;
   báo cáo thêm chỉ số chỉ tính sản phẩm mới để kết quả thuyết phục hơn.
2. **Cá nhân hoá nhóm chưa có lịch sử (34%)**: dùng tín hiệu trong phiên (mô hình theo phiên đạt 0,345 trên phiên khách)
   thay cho "thịnh hành"; cần điều chỉnh giao thức đánh giá.
3. **Thiên lệch phổ biến**: thử xếp lại có đa dạng (MMR), báo cáo đánh đổi độ chính xác – độ phủ – độ mới.
4. **Hợp nhất ALS + content-based chỉ thêm +0,006**: tinh chỉnh theo nhóm hoặc trình bày rõ ALS là phần chính.
5. **Trần candidate recall 0,527** của bộ xếp hạng lại theo phiên: cải thiện bước sinh ứng viên.
6. Demo: chế độ thuyết trình theo kịch bản; phông tải từ Google Fonts (không có mạng thì dùng phông mặc định).

## 7. Danh sách commit

| Commit | Nội dung |
|---|---|
| `394acec` | Sửa crash LightGBM, cache `meta()`, trộn danh mục cho thành viên mới, tab/biểu đồ |
| `42817b0` | Thiết kế lại: phiên từ thao tác xem, đăng ký từ phiên, khung định tuyến, hai trang |
| `4df9118` | Theme, header, lưới thẻ co giãn, khung pipeline |
| `cd733a8` | Trang Đánh giá, đủ K thẻ, thẻ cao bằng nhau, thanh bên thu gọn |
| `29dd2e1` | README: cách cài đặt hiện tại và mô tả demo mới |

