"""Streamlit demo for the hybrid recommender.

    .venv/bin/python src/build_artifacts.py data/nov_u5.parquet artifacts/
    .venv/bin/streamlit run app.py
"""
import pickle
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))  # classes must be importable to unpickle

st.set_page_config(page_title="Hệ gợi ý lai", layout="wide")


@st.cache_resource
def load():
    with open(ROOT / "artifacts" / "models.pkl", "rb") as f:
        return pickle.load(f)


A = load()
catalog: pd.DataFrame = A["catalog"]
# products that appear as an antecedent of a product-level FP-Growth rule
HAS_RULE = {next(iter(a)) for a in A["fbt"]._prod_idx if len(a) == 1}


def label(pid: int) -> str:
    r = catalog.loc[pid]
    star = "★ " if pid in HAS_RULE else ""
    return f"{star}{pid} · {r.category_code} · {r.brand} · ${r.price:,.0f}"


def show(recs, title: str):
    """recs: [(product_id, score, reason)]"""
    st.subheader(title)
    if not recs:
        st.info("Không có gợi ý.")
        return
    rows = []
    for pid, score, reason in recs:
        r = catalog.loc[pid] if pid in catalog.index else None
        rows.append({
            "product_id": pid,
            "danh mục": r.category_code if r is not None else "?",
            "thương hiệu": r.brand if r is not None else "?",
            "giá": f"${r.price:,.2f}" if r is not None else "?",
            "điểm": round(score, 3),
            "lý do": reason,
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def trending(k, exclude=()):
    return [(p, 0.0, "Đang thịnh hành") for p in A["trending"].top if p not in exclude][:k]


def guest_recs(session, k, engine):
    if not session:
        return trending(k)
    return A[engine].recommend_session(session, k)


def new_member_recs(main_cats, k):
    # placeholder for the content-based module: most popular products of each chosen
    # category, interleaved so every category gets a share of the K slots
    if not main_cats:
        return trending(k)
    per_cat = [
        catalog[catalog["main_category"] == c]["n_events"].nlargest(k).index.tolist()
        for c in sorted(main_cats)
    ]
    picks = [p for group in zip(*per_cat) for p in group]  # round-robin
    for group in per_cat:  # categories with fewer items than the others
        picks += [p for p in group if p not in picks]
    cat_of = catalog["main_category"]
    return [(p, 0.0, f"Phổ biến trong danh mục {cat_of[p]}") for p in picks[:k]]


def old_member_recs(user_id, k):
    recs = A["old_member"].recommend([user_id], k)[user_id]
    return [(p, 0.0, "Phổ biến trong danh mục bạn hay xem (tạm thời, chờ ALS)") for p in recs]


# ---------------------------------------------------------------- sidebar
st.sidebar.title("Hệ gợi ý lai")
k = st.sidebar.slider("Số gợi ý (K)", 5, 20, 10)
user_type = st.sidebar.radio("Loại người dùng", ["Khách vãng lai", "Thành viên mới", "Thành viên cũ"])
ENGINES = {
    "Bộ xếp hạng LTR (tốt nhất)": "ltr",
    "Lai: co-visitation → Item2Vec": "hybrid",
    "Co-visitation": "covisit",
    "Item2Vec": "item2vec",
}
ENGINES = {k: v for k, v in ENGINES.items() if v in A}  # artifacts built with --no-ltr have no "ltr"
engine = ENGINES[st.sidebar.selectbox("Mô hình phiên (khách)", list(ENGINES))]

# ★ products (have a "frequently bought together" rule) first, so the demo is easy to drive
products = sorted(A["demo_products"], key=lambda p: p not in HAS_RULE)
st.sidebar.caption(f"★ = có luật “thường được mua cùng” ({len(HAS_RULE & set(products))} sản phẩm)")
tab_home, tab_product, tab_cart = st.tabs(["Trang chủ", "Trang sản phẩm", "Giỏ hàng"])

# ---------------------------------------------------------------- home
with tab_home:
    if user_type == "Khách vãng lai":
        st.caption("Bộ định tuyến → mô-đun phiên (phiên rỗng → Trending)")
        session = st.multiselect("Sản phẩm đã xem trong phiên (theo thứ tự)", products, format_func=label)
        show(guest_recs(session, k, engine), "Gợi ý cho phiên hiện tại" if session else "Đang thịnh hành")
    elif user_type == "Thành viên mới":
        st.caption("Bộ định tuyến → mô-đun content-based (bản tạm: bán chạy theo danh mục)")
        cats = sorted(c for c in catalog["main_category"].unique() if c != "unknown")
        fav = st.multiselect("Danh mục yêu thích chọn lúc đăng ký", cats, default=cats[:2])
        show(new_member_recs(set(fav), k), "Gợi ý theo danh mục yêu thích")
    else:
        st.caption("Bộ định tuyến → mô-đun lọc cộng tác (bản tạm: phổ biến theo danh mục, chờ ALS)")
        uid = st.selectbox("Chọn người dùng", A["demo_users"])
        hist = A["history"][A["history"]["user_id"] == uid].sort_values("event_time")
        with st.expander(f"Lịch sử ({len(hist)} sự kiện)"):
            h = hist.tail(20).copy()
            h["sản phẩm"] = h["product_id"].map(lambda p: label(p) if p in catalog.index else p)
            st.dataframe(h[["event_time", "event_type", "sản phẩm"]], hide_index=True)
        show(old_member_recs(uid, k), "Dành riêng cho bạn")

# ---------------------------------------------------------------- product page
with tab_product:
    pid = st.selectbox("Sản phẩm đang xem", products, format_func=label)
    c1, c2 = st.columns(2)
    with c1:
        show(A["item2vec"].recommend_session([pid], k), "Sản phẩm liên quan (Item2Vec)")
    with c2:
        show(A["fbt"].recommend_items([pid], k), "Thường được mua cùng (FP-Growth)")

# ---------------------------------------------------------------- cart
with tab_cart:
    cart = st.multiselect("Sản phẩm trong giỏ", products, format_func=label, key="cart")
    if cart:
        show(A["fbt"].recommend_items(cart, k), "Mua kèm với giỏ hàng")
    else:
        st.info("Thêm sản phẩm vào giỏ để xem gợi ý mua kèm.")
