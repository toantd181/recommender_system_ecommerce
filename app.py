"""Streamlit demo for the hybrid recommender.

    .venv/bin/python src/build_artifacts.py      # fits every model, writes artifacts/models.pkl
    .venv/bin/python src/make_summary.py         # headline numbers for the 'Đánh giá' tab
    .venv/bin/streamlit run app.py

The dataset has no product names or images, so each product is shown as a card with a generated
title (brand + category), a category-coloured tile and its price. Every recommendation carries the
evidence that is actually true for it (matching attributes, bought-together confidence, whether the
user has already seen it) instead of a raw model score, which is not comparable across modules.
"""
import html
import json
import math
import pickle
import sys
from collections import Counter
from itertools import zip_longest
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))  # classes must be importable to unpickle

st.set_page_config(page_title="Hệ gợi ý lai · TMĐT", page_icon="🛍️", layout="wide")


@st.cache_resource
def load():
    with open(ROOT / "artifacts" / "models.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_resource
def derived():
    a = load()
    has_rule = {next(iter(x)) for x in a["fbt"]._prod_idx if len(x) == 1}
    pop_pct = a["catalog"]["n_events"].rank(ascending=False, pct=True)  # 0..1, small = very popular
    return has_rule, pop_pct


A = load()
catalog: pd.DataFrame = A["catalog"]
HAS_RULE, POP_PCT = derived()
SUMMARY = json.loads((ROOT / "results" / "summary.json").read_text()) if (ROOT / "results" / "summary.json").exists() else None

# curated cases that show the system well (checked against the catalog below)
EX_SESSION, EX_CART, EX_PRODUCT = [1004739, 1005104], [1004863, 5100566], 1005104
EX_USERS = [513019400, 576536641]
UNSEEN_ID = -1

ICON = {"electronics": "📱", "appliances": "🏠", "computers": "💻", "apparel": "👟", "furniture": "🛋️", "auto": "🚗",
        "construction": "🔧", "kids": "🧸", "accessories": "👜", "sport": "🏅", "medicine": "💊",
        "country_yard": "🌿", "stationery": "✏️"}
EVENT = {"view": ("👁️ xem", "gray"), "cart": ("🛒 thêm giỏ", "orange"), "purchase": ("✅ đã mua", "green")}
RANK = {"view": 1, "cart": 2, "purchase": 3}
ENGINE_NAME = {"ltr": "LightGBM re-ranker", "hybrid": "Co-visitation → Item2Vec", "covisit": "Co-visitation", "item2vec": "Item2Vec"}
MODULE_NAME = {"PopularRecent": "Thịnh hành 7 ngày", "PopularByCategory": "Phổ biến theo danh mục", "ALS": "ALS",
               "ContentBased": "Content-based"}


def module_name(key: str) -> str:
    return "ALS + content-based (hợp nhất thứ hạng)" if key.startswith("Fusion") else MODULE_NAME.get(key, key)


# ---------------------------------------------------------------- product presentation
@st.cache_resource
def _meta_cache() -> dict:
    return {}


def meta(pid: int) -> dict:
    """Display fields for a product. Cached across reruns: app.py re-executes on every click, so a
    functools cache on a function defined here would start empty each time."""
    cache = _meta_cache()
    if pid not in cache:
        cache[pid] = _meta(pid)
    return cache[pid]


def _meta(pid: int) -> dict:
    if pid not in catalog.index:
        return {"pid": pid, "title": f"Sản phẩm #{pid}", "price": 0.0, "icon": "🛍️", "hue": None, "cat": None,
                "brand": None, "leaf": None, "known": False, "cat_id": None}
    r = catalog.loc[pid]
    brand = None if r.brand in ("unknown", None) or pd.isna(r.brand) else str(r.brand).title()
    code = None if r.category_code in ("unknown", None) or pd.isna(r.category_code) else str(r.category_code)
    leaf = code.split(".")[-1].replace("_", " ").title() if code else None
    title = " · ".join(x for x in (brand, leaf) if x) or f"Sản phẩm #{pid}"
    main = code.split(".")[0] if code else None
    return {"pid": pid, "title": title, "price": float(r.price), "icon": ICON.get(main, "🛍️") if main else "🛍️",
            "hue": int(r.category_id) % 360, "cat": main, "brand": brand, "leaf": leaf, "known": code is not None,
            "cat_id": int(r.category_id)}


def esc(x) -> str:
    return html.escape(str(x))


def money(x: float) -> str:
    return f"${x:,.0f}" if x >= 100 else f"${x:,.2f}"


def popularity(pid: int) -> str:
    if pid not in POP_PCT.index:
        return ""
    return f"Top {max(1, math.ceil(POP_PCT[pid] * 100))}% phổ biến"


def tile(m: dict, height: int = 76) -> str:
    bg = f"hsl({m['hue']} 55% 55% / .22)" if m["hue"] is not None else "hsl(0 0% 50% / .14)"
    return (f"<div style='height:{height}px;border-radius:10px;background:{bg};display:flex;align-items:center;"
            f"justify-content:center;font-size:{int(height * .45)}px'>{m['icon']}</div>")


def badge_md(badges) -> str:
    return " ".join(f":{c}-badge[{t}]" for t, c in badges)


BADGE_HSL = {"green": "142 60% 36%", "blue": "215 80% 48%", "violet": "270 55% 52%", "orange": "26 90% 44%",
             "gray": "0 0% 42%", "red": "0 70% 48%"}


def badge_html(badges) -> str:
    """Inline badges that wrap instead of truncating (st.badge inside narrow columns gets cut off)."""
    out = []
    for t, c in badges:
        h = BADGE_HSL.get(c, BADGE_HSL["gray"])
        out.append(f"<span style='display:inline-block;margin:0 4px 4px 0;padding:1px 8px;border-radius:10px;font-size:12px;"
                   f"background:hsl({h} / .14);color:hsl({h});border:1px solid hsl({h} / .35)'>{esc(t)}</span>")
    return "".join(out)


def label(pid: int) -> str:
    m = meta(pid)
    return f"{'★ ' if pid in HAS_RULE else ''}{m['icon']} {m['title']} · {money(m['price'])} · #{pid}"


# ---------------------------------------------------------------- cart state
st.session_state.setdefault("cart", [])


def add_to_cart(pid: int):
    if pid not in st.session_state.cart:
        st.session_state.cart.append(pid)
        st.toast(f"Đã thêm {meta(pid)['title']} vào giỏ", icon="🛒")


def remove_from_cart(pid: int):
    st.session_state.cart = [p for p in st.session_state.cart if p != pid]


def set_state(key, value):
    st.session_state[key] = value


# ---------------------------------------------------------------- building blocks
CARD_BODY_H = 245  # same height for every card so a row of cards lines up


def card(pid: int, rank: int, badges, note: str | None, key: str):
    m = meta(pid)
    with st.container(border=True):
        st.html(tile(m))
        st.html(f"<div style='min-height:{CARD_BODY_H}px'>"
                f"<div style='font-weight:600;line-height:1.35;min-height:2.7em'>{rank}. {esc(m['title'])}</div>"
                f"<div style='font-size:26px;font-weight:700;margin:2px 0 0'>{money(m['price'])}</div>"
                f"<div style='font-size:12px;opacity:.65;margin-bottom:6px'>#{pid} · {popularity(pid)}</div>"
                f"<div>{badge_html(badges)}</div>"
                f"<div style='font-size:12px;opacity:.7;margin-top:2px'>{esc(note) if note else ''}</div></div>")
        st.button("＋ Thêm vào giỏ", key=f"add_{key}_{pid}", on_click=add_to_cart, args=(pid,), width="stretch")


def grid(items, key: str, ncols: int = 5):
    """items: [{'pid', 'badges', 'note'}]"""
    for r in range(0, len(items), ncols):
        for j, (col, it) in enumerate(zip(st.columns(ncols), items[r:r + ncols])):
            with col:
                card(it["pid"], r + j + 1, it.get("badges", []), it.get("note"), f"{key}{r}")


def row(pid: int, badges, key: str, note: str | None = None, remove: bool = False):
    m = meta(pid)
    with st.container(border=True):
        c0, c1, c2, c3 = st.columns([1, 5, 2, 2], vertical_alignment="center")
        c0.html(tile(m, 52))
        c1.html(f"<div style='font-weight:600'>{esc(m['title'])}</div>"
                f"<div style='font-size:12px;opacity:.65'>#{pid} · {popularity(pid)}</div>"
                f"<div style='margin-top:4px'>{badge_html(badges)}</div>"
                + (f"<div style='font-size:12px;opacity:.7'>{esc(note)}</div>" if note else ""))
        c2.markdown(f"**{money(m['price'])}**")
        if remove:
            c3.button("✕ Bỏ", key=f"rm_{key}_{pid}", on_click=remove_from_cart, args=(pid,))
        else:
            c3.button("＋ Giỏ", key=f"add_{key}_{pid}", on_click=add_to_cart, args=(pid,))


def section(title: str, source: str, caption: str | None = None):
    st.markdown(f"#### {title} :gray-badge[{source}]")
    if caption:
        st.caption(caption)


def pipeline(user: str, module: str, model: str):
    st.markdown(f":blue-badge[{user}] → :violet-badge[{module}] → :green-badge[{model}]")


def match_badges(pid: int, seeds) -> tuple[list, str | None]:
    """Which attributes a recommendation shares with the seed products (category / brand / price band)."""
    m, best, best_n = meta(pid), None, -1
    badges = []
    sm = [meta(s) for s in seeds if s in catalog.index]
    if m["cat_id"] is not None and any(s["cat_id"] == m["cat_id"] for s in sm):
        badges.append(("Cùng danh mục", "green"))
    if m["brand"] and any(s["brand"] == m["brand"] for s in sm):
        badges.append(("Cùng thương hiệu", "blue"))
    if m["price"] and any(s["price"] and 0.7 <= m["price"] / s["price"] <= 1.4 for s in sm):
        badges.append(("Tầm giá gần", "violet"))
    for s in sm:
        n = (s["cat_id"] == m["cat_id"]) + (s["brand"] == m["brand"] and bool(m["brand"]))
        if n > best_n:
            best, best_n = s, n
    return badges, (f"Giống {best['title']}" if best is not None and badges else None)


# ---------------------------------------------------------------- page header + user type
st.title("🛍️ Hệ gợi ý lai cho sàn thương mại điện tử")
user_type = st.segmented_control("Bạn là ai?", ["Khách vãng lai", "Thành viên mới", "Thành viên cũ"],
                                 default="Khách vãng lai", key="user_type", label_visibility="collapsed") or "Khách vãng lai"

with st.sidebar:
    st.header("Cài đặt")
    k = st.slider("Số gợi ý (K)", 5, 20, 10)
    if user_type == "Khách vãng lai":
        ENGINES = {"Bộ xếp hạng LightGBM (tốt nhất)": "ltr", "Lai: co-visitation → Item2Vec": "hybrid",
                   "Co-visitation": "covisit", "Item2Vec": "item2vec"}
        ENGINES = {n: e for n, e in ENGINES.items() if e in A}
        engine = ENGINES[st.selectbox("Mô hình theo phiên", list(ENGINES))]
    n_cart = len(st.session_state.cart)
    st.metric("🛒 Giỏ hàng", f"{n_cart} sản phẩm", f"{money(sum(meta(p)['price'] for p in st.session_state.cart))}" if n_cart else None,
              delta_color="off")
    show_unknown = st.checkbox("Hiện cả sản phẩm chưa phân loại", value=False,
                               help="Khoảng 44% sản phẩm trong danh sách chọn không có danh mục; ẩn đi cho dễ đánh giá.")
    with st.expander("Giới hạn của demo"):
        st.markdown(
            "- Kết quả offline trên tuần 24–30/11/2019, mẫu 5% người dùng; không phải A/B test.\n"
            "- Nhãn đánh giá gồm cả tương tác lặp lại, nên mô hình có xu hướng gợi ý lại sản phẩm đã xem.\n"
            "- Dữ liệu không có tên/ảnh sản phẩm; tên hiển thị được sinh từ thương hiệu + danh mục. Một phần danh mục chưa rõ.\n"
            "- 34% người dùng test chưa có lịch sử nên chỉ nhận gợi ý thịnh hành.\n"
            "- Gợi ý thiên về sản phẩm phổ biến (≈90% nằm trong top 1%, độ phủ catalog ≈2,6%).\n"
            "- “Thường được mua cùng” gồm luật khai phá thật và phần bổ sung theo danh mục; hai phần được tách rõ.")

@st.cache_resource
def build_picker(show_unknown: bool) -> list[int]:
    """Products offered in the pickers: those with a bought-together rule first, then by popularity."""
    n_events = catalog["n_events"]
    pk = [p for p in A["demo_products"] if show_unknown or meta(p)["known"]]
    return sorted(pk, key=lambda p: (p not in HAS_RULE, -int(n_events[p])))


picker = build_picker(show_unknown)
ex_session = [p for p in EX_SESSION if p in picker]
ex_cart = [p for p in EX_CART if p in catalog.index]
st.session_state.setdefault("session", [p for p in ex_session])
st.session_state["session"] = [p for p in st.session_state["session"] if p in picker]

tab_home, tab_product, tab_cart, tab_eval = st.tabs(
    ["🏠 Trang chủ", "📦 Trang sản phẩm", "🛒 Giỏ hàng", "📊 Đánh giá"], key=f"tabs_{user_type}")

@st.cache_data
def light_users_by_info() -> list[int]:
    """Few-interaction users, those whose history has known categories first (more telling in a demo)."""
    h = A["history"][A["history"]["user_id"].isin(A.get("demo_users_light", []))]
    known = h.assign(k=h["product_id"].map(lambda p: meta(p)["known"])).groupby("user_id")["k"].sum()
    return known.sort_values(ascending=False, kind="stable").index.tolist()


# ---------------------------------------------------------------- HOME
REASON_BADGE = {"Bạn đã xem trước đó": ("Xem lại", "gray"),
                "Người xem sản phẩm này cũng xem": ("Người xem cùng cũng xem", "green"),
                "Vì bạn đang xem sản phẩm tương tự": ("Tương tự (Item2Vec)", "blue"),
                "Đang thịnh hành": ("Thịnh hành", "orange")}

with tab_home:
    if user_type == "Khách vãng lai":
        with st.container(border=True):
            st.markdown("**Đầu vào · các sản phẩm đã xem trong phiên (theo thứ tự)**")
            session = st.multiselect("Phiên", picker, key="session", format_func=label,
                                     placeholder="Chọn sản phẩm đã xem…", label_visibility="collapsed")
            b1, b2, _ = st.columns([1, 1, 6])
            b1.button("Phiên mẫu", on_click=set_state, args=("session", ex_session), disabled=not ex_session)
            b2.button("Xoá phiên", on_click=set_state, args=("session", []))
            if session:
                st.markdown(" → ".join(f"{i}. {meta(p)['icon']} {meta(p)['title']}" for i, p in enumerate(session, 1)))
        if not session:
            pipeline("Khách (chưa đăng nhập)", "Phiên rỗng", "Thịnh hành")
            section("Đang thịnh hành", "Trending 7 ngày", "Chưa có sản phẩm nào đã xem nên chỉ có thể gợi ý theo độ phổ biến.")
            grid([{"pid": p, "badges": [("Thịnh hành", "orange")]} for p in A["trending"].top[:k]], "tr")
        else:
            pipeline("Khách (chưa đăng nhập)", "Gợi ý theo phiên", ENGINE_NAME[engine])
            recs = A[engine].recommend_session(session, k)
            new = [(p, r) for p, _, r in recs if r != "Bạn đã xem trước đó"]
            back = [(p, r) for p, _, r in recs if r == "Bạn đã xem trước đó"]
            section("Gợi ý cho phiên hiện tại", ENGINE_NAME[engine], "Xếp theo thứ tự phù hợp giảm dần.")
            grid([{"pid": p, "badges": [REASON_BADGE.get(r, (r, "gray"))],
                   "note": f"Sau {meta(session[-1])['title']}"} for p, r in new], "gs")
            if back:
                section("Xem lại", "Bộ xếp hạng",
                        "Khoảng 14% sản phẩm khách xem tiếp theo là sản phẩm đã xem trước đó nên mô hình xếp chúng lại.")
                grid([{"pid": p, "badges": [("Xem lại", "gray")]} for p, _ in back], "gb")

    elif user_type == "Thành viên mới":
        cats = [c for c in catalog["main_category"].value_counts().index if c != "unknown"]
        with st.container(border=True):
            st.markdown("**Đầu vào · thông tin lúc đăng ký**")
            fav = st.pills("Danh mục yêu thích", cats, selection_mode="multi", default=cats[:2],
                           format_func=lambda c: f"{ICON.get(c, '🛍️')} {c}")
            seen = st.multiselect("Sản phẩm vừa xem (tuỳ chọn, ưu tiên hơn danh mục)", picker, format_func=label,
                                  placeholder="Chọn sản phẩm…")
        seeds, via = list(seen), "sản phẩm bạn vừa xem"
        seed_groups = [seeds]
        if not seeds:
            seed_groups = [catalog[catalog["main_category"] == c]["n_events"].nlargest(3).index.tolist() for c in fav]
            seeds = [p for g in seed_groups for p in g]
            via = "sản phẩm bán chạy của từng danh mục bạn chọn (xen kẽ giữa các danh mục)"
        pipeline("Thành viên mới (ít/không có lịch sử)", "Content-based", "Cosine(danh mục · thương hiệu · tầm giá)")
        if not seeds:
            section("Đang thịnh hành", "Trending 7 ngày", "Chọn danh mục hoặc sản phẩm để cá nhân hoá.")
            grid([{"pid": p, "badges": [("Thịnh hành", "orange")]} for p in A["trending"].top[:k]], "nt")
        else:
            section("Gợi ý cho thành viên mới", "Content-based", f"Hồ sơ được dựng từ {via}; điểm cộng cho sản phẩm đang phổ biến.")
            items = []
            # a single averaged profile lets the biggest category take every slot, and the popularity
            # prior pulls electronics into small categories (stationery alone got 0/10 stationery);
            # so rank each chosen category separately, keep its own products first, then take turns.
            # The model only scores its candidate pool (5,000 popular recent products, which holds 0-14
            # items of the small categories), so a category it cannot fill is topped up with that
            # category's best sellers from the full catalog.
            def ranked(group, cat_=None):
                pool = A["content"].cand_ids
                r = [p for p, _ in A["content"].recommend_from_items(group, len(pool), exclude=seeds)]
                if cat_ is None:
                    return r[:k]
                own = [p for p in r if meta(p)["cat"] == cat_]
                if len(own) < k:
                    best = catalog[catalog["main_category"] == cat_]["n_events"].nlargest(k + len(seeds)).index
                    own += [p for p in best if p not in seeds and p not in own][:k - len(own)]
                rest = [p for p in r if meta(p)["cat"] != cat_]
                return (own + rest)[:k]
            lists = [ranked(g, c) for g, c in zip(seed_groups, fav if not seen else [None]) if g]
            merged = list(dict.fromkeys(p for tier in zip_longest(*lists) for p in tier if p is not None))[:k]
            for p in merged:
                badges, note = match_badges(p, seeds)
                items.append({"pid": p, "badges": badges or [("Phổ biến", "orange")], "note": note})
            grid(items, "nm")

    else:
        router = A["router"]
        heavy = [u for u in EX_USERS + A["demo_users"] if u in A["demo_users"]]
        heavy = list(dict.fromkeys(heavy))[:40]
        light = light_users_by_info()[:20]
        kind = {**{u: "Có nhiều lịch sử" for u in heavy}, **{u: "Ít lịch sử" for u in light}, UNSEEN_ID: "Chưa có lịch sử"}
        with st.container(border=True):
            st.markdown("**Đầu vào · chọn người dùng**")
            c1, c2 = st.columns([3, 2])
            uid = c1.selectbox("Người dùng", list(kind), label_visibility="collapsed",
                               format_func=lambda u: f"{kind[u]} · {'người dùng mới' if u == UNSEEN_ID else u}")
            hide_bought = c2.toggle("Ẩn sản phẩm đã mua", value=True)
        group = router.group_of([uid])[uid]
        pipeline(f"Thành viên · nhóm {group}", "Bộ định tuyến", module_name(A["routing"][group]))
        hist = A["history"][A["history"]["user_id"] == uid].sort_values("event_time")
        status: dict[int, int] = {}
        for p, e in zip(hist["product_id"], hist["event_type"]):
            status[p] = max(status.get(p, 0), RANK[e])
        if len(hist):
            with st.expander(f"Lịch sử của người dùng ({len(hist)} sự kiện)", expanded=True):
                tops = Counter(meta(p)["leaf"] or meta(p)["brand"] or "chưa rõ" for p in hist["product_id"].unique())
                st.markdown("**Hay quan tâm:** " + ", ".join(f"{n} ({c})" for n, c in tops.most_common(3)))
                st.markdown(" ".join(f":{EVENT[e][1]}-badge[{EVENT[e][0]} · {meta(p)['title']}]"
                                     for p, e in list(zip(hist["product_id"], hist["event_type"]))[-8:]))
        cat_counts = Counter(meta(p)["cat_id"] for p in status)
        if group == "unseen":
            section("Đang thịnh hành", "Trending 7 ngày", "Người dùng chưa có lịch sử nên bộ định tuyến chỉ có thể dùng độ phổ biến.")
            grid([{"pid": p, "badges": [("Thịnh hành", "orange")]} for p in A["trending"].top[:k]], "ou")
        else:
            recs = router.recommend([uid], k + 15)[uid]
            if hide_bought:
                recs = [p for p in recs if status.get(p, 0) < 3]
            recs = recs[:k]
            items, n_new = [], 0
            for p in recs:
                s = status.get(p, 0)
                badges = [(("Đã mua", "green") if s == 3 else ("Đã thêm giỏ", "orange") if s == 2
                           else ("Đã xem", "gray") if s == 1 else ("Mới", "blue"))]
                n_new += s == 0
                same = cat_counts.get(meta(p)["cat_id"], 0)
                items.append({"pid": p, "badges": badges,
                              "note": f"Cùng danh mục với {same} sản phẩm bạn đã xem" if same else "Phổ biến & hợp hồ sơ"})
            section("Dành riêng cho bạn", "ALS + content-based", f"{n_new}/{len(recs)} gợi ý là sản phẩm mới với người dùng này; "
                    "phần còn lại là sản phẩm họ đã tương tác (mô hình được thưởng khi gợi ý lại, xem mục Giới hạn).")
            grid(items, "om")

# ---------------------------------------------------------------- PRODUCT PAGE
def fbt_blocks(query, key, k_=10):
    """Split FP-Growth output: mined rules (true 'bought together') vs category/popularity filler."""
    res = A["fbt"].recommend_items(query, k_)
    rules = [(p, s - 2.0) for p, s, r in res if r == "Thường được mua cùng"]
    filler = [(p, r) for p, s, r in res if r != "Thường được mua cùng"]
    return rules, filler


with tab_product:
    options = picker if EX_PRODUCT not in picker else [EX_PRODUCT] + [p for p in picker if p != EX_PRODUCT]
    pid = st.selectbox("Sản phẩm đang xem", options, format_func=label, key="pid")
    m = meta(pid)
    with st.container(border=True):
        c0, c1, c2 = st.columns([1, 5, 2], vertical_alignment="center")
        c0.html(tile(m, 96))
        c1.markdown(f"### {m['title']}")
        tags = [(m["leaf"] or "Chưa rõ danh mục", "gray")] + ([("★ Có luật mua cùng", "green")] if pid in HAS_RULE else [])
        c1.markdown(badge_md(tags) + f"  \n`#{pid}` · {popularity(pid)}")
        c2.markdown(f"## {money(m['price'])}")
        c2.button("＋ Thêm vào giỏ", key=f"add_hero_{pid}", on_click=add_to_cart, args=(pid,), type="primary")
    left, right = st.columns(2)
    with left:
        section("Sản phẩm tương tự", "Item2Vec", "Cùng ngữ cảnh xem trong phiên: thường là sản phẩm thay thế.")
        for p, s, _ in A["item2vec"].recommend_session([pid], k):
            row(p, [(f"Tương đồng {s:.2f}", "blue")], f"i2v{pid}")
    with right:
        rules, filler = fbt_blocks([pid], "p", k)
        section("Thường được mua cùng", "FP-Growth", "Chỉ gồm luật khai phá từ các giỏ hàng thật.")
        if rules:
            for p, conf in rules:
                row(p, [(f"{conf:.0%} khách mua sản phẩm này cũng mua", "green")], f"fb{pid}")
        else:
            st.info("Sản phẩm này chưa có luật mua cùng đủ mạnh. Hãy chọn sản phẩm có dấu ★.")
        if filler:
            with st.expander(f"Bổ sung (không phải luật mua cùng) · {len(filler)} sản phẩm"):
                st.caption("Dùng để lấp chỗ khi chưa đủ luật: sản phẩm phổ biến cùng danh mục hoặc đang thịnh hành.")
                for p, r in filler[:5]:
                    row(p, [(r, "gray")], f"ff{pid}")

# ---------------------------------------------------------------- CART
with tab_cart:
    cart = st.session_state.cart
    left, right = st.columns([3, 4])
    with left:
        section("Giỏ hàng", f"{len(cart)} sản phẩm")
        add_pick = st.selectbox("Thêm sản phẩm", [None] + picker, format_func=lambda p: "Chọn sản phẩm…" if p is None else label(p),
                                key="cart_pick", label_visibility="collapsed")
        b1, b2, b3 = st.columns(3)
        b1.button("Thêm", on_click=lambda: add_to_cart(st.session_state.cart_pick) if st.session_state.cart_pick else None, width="stretch")
        b2.button("Giỏ mẫu", on_click=set_state, args=("cart", list(ex_cart)), width="stretch", disabled=not ex_cart)
        b3.button("Xoá giỏ", on_click=set_state, args=("cart", []), width="stretch")
        if cart:
            for p in cart:
                row(p, [], "cart", remove=True)
            st.metric("Tổng tiền", money(sum(meta(p)["price"] for p in cart)))
        else:
            st.info("Giỏ hàng trống. Bấm **Giỏ mẫu** để thử ngay, hoặc thêm sản phẩm từ các trang khác.")
    with right:
        section("Mua kèm với giỏ hàng", "FP-Growth", "Luật khai phá từ giỏ hàng thật; loại sản phẩm đã có trong giỏ.")
        if cart:
            rules, filler = fbt_blocks(cart, "c", k)
            if rules:
                for p, conf in rules:
                    row(p, [(f"{conf:.0%} khách mua sản phẩm trong giỏ cũng mua", "green")], "cr")
            else:
                st.info("Chưa có luật mua kèm đủ mạnh cho các sản phẩm này; thử thêm sản phẩm có dấu ★.")
            if filler:
                with st.expander(f"Bổ sung (không phải luật mua cùng) · {len(filler)} sản phẩm"):
                    for p, r in filler[:5]:
                        row(p, [(r, "gray")], "cf")
        else:
            st.caption("Thêm sản phẩm vào giỏ để xem gợi ý mua kèm.")

# ---------------------------------------------------------------- EVALUATION
with tab_eval:
    if not SUMMARY:
        st.info("Chưa có `results/summary.json`. Chạy `python src/make_summary.py`.")
    else:
        st.caption(f"Kết quả offline trên tuần test {SUMMARY['test_window']} (mẫu 5% người dùng). "
                   "Mọi tham số được chọn trên tuần validation; test chạy đúng một lần. Khoảng tin cậy 95% bootstrap.")
        g, ci = SUMMARY["router"]["groups"], SUMMARY["router"]["ci"]
        section("Gợi ý cho người dùng đăng nhập", "Router vs baseline", f"{g['Router']['all']['n_users']:,} người dùng có thêm giỏ/mua ở tuần test.")
        names = ["PopularByCategory", "ContentBased", "ALS", "Router"]
        st.dataframe(pd.DataFrame({
            "Mô hình": [("★ " if n == "Router" else "") + n for n in names],
            "HitRate@10": [f"{ci[n]['hit@10'][0]:.3f} [{ci[n]['hit@10'][1]:.3f}; {ci[n]['hit@10'][2]:.3f}]" for n in names],
            "NDCG@10": [f"{ci[n]['ndcg@10'][0]:.3f}" for n in names]}), hide_index=True, width="stretch")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**HitRate@10 theo nhóm người dùng**")
            grp = pd.DataFrame({n: {k_: g[n][k_]["hit@10"] for k_ in ("unseen", "light", "heavy")} for n in names})
            grp.index = ["1. Chưa có lịch sử", "2. Ít lịch sử (1–4)", "3. Nhiều lịch sử (≥5)"]
            st.bar_chart(grp, stack=False, horizontal=True, sort=False)
        with c2:
            st.markdown("**Bộ định tuyến chọn mô-đun**")
            st.table(pd.DataFrame({"Nhóm": ["Chưa có lịch sử", "Ít lịch sử", "Nhiều lịch sử"],
                                   "Mô-đun": [module_name(SUMMARY["routing"][x]) for x in ("unseen", "light", "heavy")],
                                   "Người dùng": [f"{g['Router'][x]['n_users']:,}" for x in ("unseen", "light", "heavy")]}))
        c3, c4 = st.columns(2)
        with c3:
            section("Khách vãng lai", "đoán sản phẩm tiếp theo trong phiên")
            st.dataframe(pd.DataFrame({"Mô hình": list(SUMMARY["session"]["hit@10"]),
                                       "HitRate@10": list(SUMMARY["session"]["hit@10"].values())}), hide_index=True, width="stretch")
            st.caption(f"Trần của mọi bộ xếp hạng lại (candidate recall): {SUMMARY['session']['candidate_recall']:.3f}")
        with c4:
            section("Thường được mua cùng", "leave-one-out trên giỏ hàng test")
            st.dataframe(pd.DataFrame({"Hệ thống": list(SUMMARY["fbt"]), "HitRate@10": list(SUMMARY["fbt"].values())}),
                         hide_index=True, width="stretch")
