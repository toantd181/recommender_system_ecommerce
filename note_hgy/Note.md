## Chủ đề
Hệ thống gợi ý sản phẩm lai (Hybrid) cho sàn thương mại điện tử theo từng nhóm khách hàng 
# Bài toán thực tế và mục đích
- Một sàn thương mại điện tử (TMĐT) đa ngành hàng có hàng trăm nghìn sản phẩm; người dùng khó tự tìm được món phù hợp, còn sàn mất cơ hội bán hàng. 
- Khó khăn chính là mỗi nhóm người dùng có lượng dữ liệu rất khác nhau: 
	- Khách vãng lai chưa đăng nhập chỉ có vài cú click trong phiên hiện tại 
	- Thành viên mới gần như chưa có lịch sử (cold-start)
	- Thành viên cũ có lịch sử tương tác dày. 
	- Một thuật toán duy nhất không phục vụ tốt cả ba nhóm.  
- Mục đích: xây dựng một hệ gợi ý lai, tự phân luồng người dùng và chọn kỹ thuật phù hợp cho từng nhóm, kèm gợi ý “thường được mua cùng” ở trang sản phẩm và giỏ hàng, nhằm tăng tỉ lệ click, thêm vào giỏ và giá trị đơn hàng. 
- Yêu cầu:  
	1. Trả về Top-K sản phẩm cho mọi loại người dùng, kể cả khi không có lịch sử 
	2. Tận dụng phản hồi ngầm (implicit feedback) vì dữ liệu không có điểm đánh giá
	3. Đánh giá được bằng độ đo offline
	4. Có bản demo minh họa. 
	
# Dữ liệu
- Bộ eCommerce behavior data from multi category store (Kaggle, REES46): nhật ký hành vi của một sàn TMĐT đa ngành hàng, hơn 100 triệu sự kiện trong tháng 10–11/2019. 
	- https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store
	- Mỗi dòng gồm event_time, event_type, product_id, category_id, category_code, brand, price, user_id, user_session. 
	- Do dữ liệu lớn, nhóm sẽ lấy mẫu (ví dụ 1 tháng hoặc một số ngành hàng) và chia train/test theo thời gian. 
- Chấm điểm sự kiện cho implicit CF: mỗi cặp (người dùng, sản phẩm) được cộng dồn điểm theo trọng số dưới đây (có thể nhân thêm hệ số suy giảm theo thời gian), sau đó dùng làm độ tin cậy $c = 1 + α·r$ cho ALS. 

| Sự kiện            |     view     |     cart      | remove_from_cart |      purchase      |
| :----------------- | :----------: | :-----------: | :--------------: | :----------------: |
| Trọng số (đề xuất) |      1       |       3       |        -2        |         5          |
| Ý nghĩa            | Quan tâm nhẹ | Ý định mua rõ |  Giảm quan tâm   | Tín hiệu mạnh nhất |

# Phương pháp đề xuất 
- Hệ thống gồm một bộ định tuyến và bốn mô-đun gợi ý: 

| Nhóm người dùng                         | Nhân diện                                     | Phương pháp                                                                                                                                                                                                               | Ví dụ kết quả                                                 |
| :-------------------------------------- | :-------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :------------------------------------------------------------ |
| **Khách vãng lai (Guest)**              | Chưa đăng nhập, chỉ có phiên hiện tại         | Gợi ý dựa trên phiên: Item2Vec (Word2Vec trên chuỗi sản phẩm trong phiên), nâng cao với SR-GNN. Phiên rỗng → Trending (độ phổ biến theo cửa sổ trượt 24 giờ / 7 ngày).                                                    | Đang xem điện thoại → gợi ý ốp lưng, tai nghe                 |
| **Thành viên mới (New member)**         | Đã đăng nhập, số tương tác < ngưỡng (ví dụ 5) | Gợi ý dựa trên nội dung: vector đặc trưng sản phẩm (category_code, brand, khoảng giá), hồ sơ người dùng từ danh mục yêu thích chọn lúc đăng ký; xếp hạng bằng Cosine Similarity, kết hợp sản phẩm bán chạy theo danh mục. | Chọn “điện tử, thời trang” → top bán chạy trong 2 danh mục    |
| **Thành viên cũ (Old member)**          | Đã đăng nhập, lịch sử tương tác dày           | Lọc cộng tác: Alternating Least Squares cho implicit feedback trên ma trận điểm người dùng × sản phẩm.                                                                                                                    | Người hay mua đồ bếp → sản phẩm mà người dùng tương tự đã mua |
| **Mọi người dùng (trang SP, giỏ hàng)** | Đang xem sản phẩm / có giỏ hàng               | Market Basket Analysis: FP-Growth trên giỏ hàng (các sản phẩm purchase cùng phiên) → luật kết hợp “Frequently Bought Together”.                                                                                           | Giỏ có máy ảnh → thẻ nhớ, túi đựng                            |
  
 -  Đánh giá: 
	 - Chia dữ liệu theo thời gian; dùng Precision@K, Recall@K, NDCG@K, HitRate@K/MRR (cho gợi ý phiên) 
	 - So sánh với baseline “phổ biến nhất”. 
	 - Luật kết hợp đánh giá bằng support, confidence, lift. 
# Đầu vào và đầu ra của hệ thống 
|                       | Nội dung                                                                                                                                                                                                                       |
| :-------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Đầu vào (offline)** | File log sự kiện CSV (user_id, user_session, product_id, event_type, event_time, category_code, brand, price) → tiền xử lý, chấm điểm, huấn luyện các mô hình.                                                                 |
| **Đầu vào (online)**  | Ngữ cảnh yêu cầu: user_id (có thể rỗng), chuỗi sản phẩm đã xem trong phiên, sản phẩm đang xem / giỏ hàng hiện tại, danh mục yêu thích (thành viên mới), K                                                                      |
| **Đầu ra**            | Danh sách Top-K sản phẩm: product_id, danh mục, thương hiệu, giá, điểm gợi ý và lý do gợi ý (“Vì bạn đang xem…”, “Đang thịnh hành”, “Thường được mua cùng”, “Người dùng tương tự cũng thích”); kèm báo cáo các độ đo đánh giá. |
# Kịch bản ứng dụng và công cụ 
- Trang chủ: guest thấy Trending; thành viên mới thấy gợi ý theo danh mục yêu thích; thành viên cũ thấy “Dành riêng cho bạn” (ALS). 
- Trang chi tiết sản phẩm: “Sản phẩm liên quan” (Item2Vec) và “Thường được mua cùng” (FP-Growth). 
- Giỏ hàng: gợi ý mua kèm từ luật kết hợp. 
- Công cụ: Python, Pandas/PySpark, Gensim (Word2Vec), PyTorch (SR-GNN), implicit (ALS), scikit-learn (Cosine), mlxtend/Spark MLlib (FP-Growth), Streamlit (demo).