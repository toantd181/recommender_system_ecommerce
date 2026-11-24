"""Streamlit demo for the hybrid recommender.

    .venv/bin/python src/build_artifacts.py      # fits every model, writes artifacts/models.pkl
    .venv/bin/python src/make_summary.py         # headline numbers for the 'Đánh giá' page
    .venv/bin/streamlit run app.py

Two pages: a shop and the offline evaluation. In the shop the visitor browses like on a real site:
"Xem" opens a quick view and appends the product to the session trail, and every shelf re-ranks from
that trail. The same trail follows the visitor when they sign up (guest -> new member), which is the
cold-start hand-off the hybrid system exists for. A panel shows what the system did for the current
user: their group, the router's decision, the module and where the candidates came from.

The dataset has no product names or images, so each product is shown with a generated title (brand +
category), a category-coloured tile and its price. Every recommendation carries the evidence that is
actually true for it instead of a raw model score, which is not comparable across modules.
"""
import html
import json
import math
import pickle
import sys
import time
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

# curated cases that show the system well
EX_SESSION, EX_CART = [1004739, 1005104], [1004863, 5100566]
EX_USERS = [513019400, 576536641]
UNSEEN_ID = -1
PERSONAS = ["Khách vãng lai", "Thành viên mới", "Thành viên cũ"]
GUEST, NEW, MEMBER = PERSONAS

ICON = {"electronics": "📱", "appliances": "🏠", "computers": "💻", "apparel": "👟", "furniture": "🛋️", "auto": "🚗",
        "construction": "🔧", "kids": "🧸", "accessories": "👜", "sport": "🏅", "medicine": "💊",
        "country_yard": "🌿", "stationery": "✏️"}
EVENT = {"view": ("👁️ xem", "gray"), "cart": ("🛒 thêm giỏ", "orange"), "purchase": ("✅ đã mua", "green")}
RANK = {"view": 1, "cart": 2, "purchase": 3}
ENGINE_NAME = {"ltr": "LightGBM re-ranker", "hybrid": "Co-visitation → Item2Vec", "covisit": "Co-visitation", "item2vec": "Item2Vec"}
MODULE_NAME = {"PopularRecent": "Thịnh hành 7 ngày", "PopularByCategory": "Phổ biến theo danh mục", "ALS": "ALS",
               "ContentBased": "Content-based"}
SOURCE_NAME = {"ALSRecommender": "ALS", "ContentBased": "Content"}
REASON_BADGE = {"Bạn đã xem trước đó": ("Xem lại", "gray"),
                "Người xem sản phẩm này cũng xem": ("Người xem cùng cũng xem", "green"),
                "Vì bạn đang xem sản phẩm tương tự": ("Tương tự (Item2Vec)", "blue"),
                "Đang thịnh hành": ("Thịnh hành", "orange")}
RULE = "Thường được mua cùng"


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


def tile(m: dict, height: int = 64) -> str:
    bg = f"hsl({m['hue']} 55% 55% / .22)" if m["hue"] is not None else "hsl(0 0% 50% / .14)"
    return (f"<div style='height:{height}px;border-radius:10px;background:{bg};display:flex;align-items:center;"
            f"justify-content:center;font-size:{int(height * .5)}px'>{m['icon']}</div>")


BADGE_HSL = {"green": "142 60% 36%", "blue": "215 80% 48%", "violet": "270 55% 52%", "orange": "26 90% 44%",
             "gray": "0 0% 42%", "red": "0 70% 48%"}


def badge_html(badges) -> str:
    """Inline badges that wrap instead of truncating (st.badge inside narrow columns gets cut off)."""
    out = []
    for t, c in badges:
        h = BADGE_HSL.get(c, BADGE_HSL["gray"])
        out.append(f"<span style='display:inline-block;margin:0 4px 4px 0;padding:1px 7px;border-radius:10px;font-size:11.5px;"
                   f"background:hsl({h} / .14);color:hsl({h});border:1px solid hsl({h} / .35)'>{esc(t)}</span>")
    return "".join(out)


def label(pid: int) -> str:
    m = meta(pid)
    return f"{'★ ' if pid in HAS_RULE else ''}{m['icon']} {m['title']} · {money(m['price'])} · #{pid}"


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


@st.cache_resource
def build_picker(show_unknown: bool) -> list[int]:
    """Products offered in the search box: those with a bought-together rule first, then by popularity."""
    n_events = catalog["n_events"]
    pk = [p for p in A["demo_products"] if show_unknown or meta(p)["known"]]
    return sorted(pk, key=lambda p: (p not in HAS_RULE, -int(n_events[p])))


# ---------------------------------------------------------------- state + actions
ss = st.session_state
ss.setdefault("persona", GUEST)
ss.setdefault("trail", list(EX_SESSION))   # products viewed in this visit, in order
ss.setdefault("cart", [])
ss.setdefault("qv", None)                  # product open in the quick view
ss.setdefault("search_open", False)
ss.setdefault("from_guest", 0)             # >0: signed up from a guest session of this many views


def view(pid: int):
    """Open the quick view; a view is also what the session models read."""
    if not ss.trail or ss.trail[-1] != pid:
        ss.trail.append(pid)
    ss.qv = pid


def close_qv():
    ss.qv = None


def add_to_cart(pid: int):
    if pid not in ss.cart:
        ss.cart.append(pid)
        # no st.toast here: a callback that renders during a fragment/dialog rerun corrupts the page tree
        ss.flash = f"Đã thêm {meta(pid)['title']} vào giỏ"


def show_flash():
    msg = ss.pop("flash", None)
    if msg:
        st.toast(msg, icon="🛒")


def remove_from_cart(pid: int):
    ss.cart = [p for p in ss.cart if p != pid]


def drop_view(i: int):
    ss.trail = ss.trail[:i] + ss.trail[i + 1:]


def set_state(key, value):
    ss[key] = value


def pick_from_search():
    pid = ss.get("search_pick")
    if pid is not None:
        view(pid)
    ss.search_open = False
    ss.search_pick = None


def sign_up():
    """Guest -> new member: favourite categories and profile come from what they browsed."""
    cats = [meta(p)["cat"] for p in ss.trail if meta(p)["cat"]]
    ss.fav = list(dict.fromkeys(cats))[:3]
    ss.from_guest = len(ss.trail)
    ss.persona = NEW


def sign_out():
    ss.persona = GUEST
    ss.from_guest = 0


# ---------------------------------------------------------------- building blocks
def section(title: str, source: str, caption: str | None = None):
    st.markdown(f"#### {title} :gray-badge[{source}]")
    if caption:
        st.caption(caption)


def card(pid: int, badges, note: str | None, key: str, rank: int):
    m = meta(pid)
    with st.container(border=True, width=212):
        st.html(tile(m)
                + f"<div style='min-height:118px;margin-top:6px'>"
                  f"<div style='font-weight:600;font-size:14px;line-height:1.3'>{rank}. {esc(m['title'])}</div>"
                  f"<div style='font-size:21px;font-weight:700;margin:1px 0'>{money(m['price'])}</div>"
                  f"<div style='font-size:11.5px;opacity:.6;margin-bottom:4px'>#{pid} · {popularity(pid)}</div>"
                  f"<div>{badge_html(badges)}</div>"
                  + (f"<div style='font-size:11.5px;opacity:.7'>{esc(note)}</div>" if note else "") + "</div>")
        with st.container(horizontal=True, gap="small"):
            st.button("Xem", icon=":material/visibility:", key=f"v_{key}_{pid}", on_click=view, args=(pid,),
                      width="stretch")
            st.button("", icon=":material/add_shopping_cart:", key=f"c_{key}_{pid}", on_click=add_to_cart,
                      args=(pid,), help="Thêm vào giỏ")


def shelf(items, key: str):
    """items: [{'pid', 'badges', 'note'}] -> cards that wrap to the width of the page."""
    with st.container(horizontal=True, wrap=True, gap="small"):
        for i, it in enumerate(items, 1):
            card(it["pid"], it.get("badges", []), it.get("note"), key, i)


def mini_row(pid: int, badges, key: str, action: str = "view"):
    """Compact row for side panels and the quick view."""
    m = meta(pid)
    with st.container(horizontal=True, vertical_alignment="center", gap="small", wrap=False):
        st.html(f"<div style='display:flex;gap:10px;align-items:center'><div style='width:42px;flex:none'>{tile(m, 42)}</div>"
                f"<div><div style='font-weight:600;font-size:13.5px'>{esc(m['title'])} · {money(m['price'])}</div>"
                f"<div>{badge_html(badges)}</div></div></div>", width="stretch")
        if action == "view":
            st.button("", icon=":material/visibility:", key=f"mv_{key}_{pid}", on_click=view, args=(pid,), help="Xem")
        elif action == "cart":
            st.button("", icon=":material/add_shopping_cart:", key=f"mc_{key}_{pid}", on_click=add_to_cart, args=(pid,),
                      help="Thêm vào giỏ")
        elif action == "remove":
            st.button("", icon=":material/close:", key=f"mr_{key}_{pid}", on_click=remove_from_cart, args=(pid,),
                      help="Bỏ khỏi giỏ")


def explain(steps, ms: float | None):
    """'What did the system do' panel: user -> router -> module -> candidates."""
    if not ss.get("explain", True):
        return
    with st.container(border=True):
        with st.container(horizontal=True, gap="medium", vertical_alignment="top"):
            for i, (head, body) in enumerate(steps):
                if i:
                    st.markdown("<div style='opacity:.4;font-size:20px;padding-top:6px'>→</div>", unsafe_allow_html=True,
                                width="content")
                st.markdown(f"<div style='font-size:12px;opacity:.65'>{esc(head)}</div>"
                            f"<div style='font-size:14px;font-weight:600;line-height:1.35'>{body}</div>",
                            unsafe_allow_html=True)
        if ms is not None:
            st.caption(f"⏱ {ms:.0f} ms để tạo gợi ý")


def fbt_split(query, k_=10):
    """FP-Growth output: mined rules (true 'bought together') vs category/popularity filler."""
    res = A["fbt"].recommend_items(query, k_)
    return ([(p, s - 2.0) for p, s, r in res if r == RULE], [(p, r) for p, s, r in res if r != RULE])


# ---------------------------------------------------------------- cached model calls (time measured on the miss)
@st.cache_data(max_entries=256, show_spinner=False)
def session_recs(engine: str, trail: tuple, k: int):
    t = time.perf_counter()
    recs = A[engine].recommend_session(list(trail), k)
    return recs, (time.perf_counter() - t) * 1000


@st.cache_data(max_entries=256, show_spinner=False)
def member_recs(uid: int, n: int):
    """Router output plus, for a fusion module, each member model's rank of every item."""
    t = time.perf_counter()
    router = A["router"]
    group = router.group_of([uid])[uid]
    recs = router.recommend([uid], n)[uid]
    ms = (time.perf_counter() - t) * 1000
    module = router.modules[group]
    ranks = {}
    for m in getattr(module, "models", []):
        lst = m.recommend([uid], getattr(module, "depth", 50)).get(uid, [])
        ranks[SOURCE_NAME.get(type(m).__name__, type(m).__name__)] = {p: r for r, p in enumerate(lst, 1)}
    return recs, ranks, ms


@st.cache_data(max_entries=256, show_spinner=False)
def new_member_recs(seed_groups: tuple, cats: tuple, k: int):
    """Content-based for a profile with no history. A single averaged profile lets the biggest category take
    every slot and its popularity prior pulls electronics into small categories (stationery alone got 0/10
    stationery), so each chosen category is ranked on its own, keeps its own products first, and the lists
    take turns. The model only scores a 5,000-product candidate pool (0-14 items of the small categories),
    so a category it cannot fill is topped up with that category's best sellers."""
    t = time.perf_counter()
    seeds = [p for g in seed_groups for p in g]
    pool = A["content"].cand_ids

    def ranked(group, cat_):
        r = [p for p, _ in A["content"].recommend_from_items(list(group), len(pool), exclude=seeds)]
        if cat_ is None:
            return r[:k]
        own = [p for p in r if meta(p)["cat"] == cat_]
        if len(own) < k:
            best = catalog[catalog["main_category"] == cat_]["n_events"].nlargest(k + len(seeds)).index
            own += [p for p in best if p not in seeds and p not in own][:k - len(own)]
        return (own + [p for p in r if meta(p)["cat"] != cat_])[:k]

    lists = [ranked(g, c) for g, c in zip(seed_groups, cats) if g]
    merged = list(dict.fromkeys(p for tier in zip_longest(*lists) for p in tier if p is not None))[:k]
    return merged, (time.perf_counter() - t) * 1000


@st.cache_data
def light_users_by_info() -> list[int]:
    """Few-interaction users, those whose history has known categories first (more telling in a demo)."""
    h = A["history"][A["history"]["user_id"].isin(A.get("demo_users_light", []))]
    known = h.assign(k=h["product_id"].map(lambda p: meta(p)["known"])).groupby("user_id")["k"].sum()
    return known.sort_values(ascending=False, kind="stable").index.tolist()


# ---------------------------------------------------------------- shop sections
def trail_bar(picker):
    n = len(ss.trail)
    with st.container(border=True):
        with st.container(horizontal=True, vertical_alignment="center", gap="small"):
            st.markdown(f"**🧭 Phiên đang xem** · {n} lượt xem", width="stretch")
            st.button("Tìm sản phẩm", icon=":material/search:", type="tertiary",
                      on_click=set_state, args=("search_open", not ss.search_open))
            st.button("Phiên mẫu", icon=":material/auto_awesome:", type="tertiary", on_click=set_state,
                      args=("trail", list(EX_SESSION)))
            st.button("Xoá phiên", icon=":material/delete:", type="tertiary", on_click=set_state, args=("trail", []),
                      disabled=not n)
        if ss.search_open:  # 12k options: only sent to the browser while the search box is open
            st.selectbox("Tìm sản phẩm", picker, index=None, key="search_pick", format_func=label,
                         on_change=pick_from_search, placeholder="Gõ thương hiệu, danh mục hoặc mã sản phẩm… (★ = có luật mua cùng)",
                         label_visibility="collapsed")
        if not n:
            st.caption("Bấm **Xem** trên một sản phẩm bất kỳ để bắt đầu phiên; mỗi lượt xem làm gợi ý thay đổi ngay.")
            return
        start = max(0, n - 8)
        with st.container(horizontal=True, wrap=True, gap="small", vertical_alignment="center"):
            if start:
                st.caption(f"+{start} lượt trước")
            for i in range(start, n):
                m = meta(ss.trail[i])
                if i > start:
                    st.markdown("<span style='opacity:.4'>→</span>", unsafe_allow_html=True, width="content")
                st.button(f"{m['icon']} {m['title']} · {money(m['price'])}", key=f"chip_{i}_{ss.trail[i]}",
                          icon=":material/close:", icon_position="right", on_click=drop_view, args=(i,),
                          help="Bỏ lượt xem này khỏi phiên", type="primary" if i == n - 1 else "secondary")


def session_shelf(k: int, engine: str, title: str | None = None):
    """Next-item recommendations from the trail. Returns (steps, ms) for the explain panel."""
    recs, ms = session_recs(engine, tuple(ss.trail), k)
    new = [(p, r) for p, _, r in recs if r != "Bạn đã xem trước đó"]
    back = [(p, r) for p, _, r in recs if r == "Bạn đã xem trước đó"]
    last = meta(ss.trail[-1])
    section(title or "Gợi ý cho phiên hiện tại", ENGINE_NAME[engine], f"Sau {last['title']} · xếp theo mức phù hợp giảm dần.")
    shelf([{"pid": p, "badges": [REASON_BADGE.get(r, (r, "gray"))]} for p, r in new], "s")
    if back:
        section("Xem lại", ENGINE_NAME[engine],
                "Khoảng 14% sản phẩm khách xem tiếp theo là sản phẩm đã xem trước đó nên mô hình xếp chúng lại.")
        shelf([{"pid": p, "badges": [("Xem lại", "gray")]} for p, _ in back], "b")
    src = Counter(REASON_BADGE.get(r, (r,))[0] for _, _, r in recs)
    return src, ms


def trending_shelf(k: int, why: str):
    section("Đang thịnh hành", "Trending 7 ngày", why)
    shelf([{"pid": p, "badges": [("Thịnh hành", "orange")]} for p in A["trending"].top[:k]], "t")


def guest_home(k: int, engine: str):
    if not ss.trail:
        explain([("Người dùng", "Khách · phiên rỗng"), ("Định tuyến", "Không có tín hiệu nào"),
                  ("Mô-đun", "Thịnh hành 7 ngày"), ("Ứng viên", f"{k} sản phẩm phổ biến nhất")], None)
        trending_shelf(k, "Chưa xem sản phẩm nào nên chỉ có thể gợi ý theo độ phổ biến.")
        return
    holder = st.empty()  # the panel sits above the shelves but needs their numbers
    src, ms = session_shelf(k, engine)
    with holder.container():
        explain([("Người dùng", f"Khách · {len(ss.trail)} lượt xem"), ("Định tuyến", "Chưa đăng nhập → gợi ý theo phiên"),
                 ("Mô-đun", ENGINE_NAME[engine]),
                 ("Ứng viên", " · ".join(f"{n} {s.lower()}" for s, n in src.most_common()))], ms)


def new_member_home(k: int, engine: str):
    cats = [c for c in catalog["main_category"].value_counts().index if c != "unknown"]
    if ss.from_guest:
        st.success(f"Chào thành viên mới! Hồ sơ được dựng từ {ss.from_guest} sản phẩm bạn đã xem lúc còn là khách.",
                   icon="🎉")
    ss.setdefault("fav", cats[:2])
    fav = st.pills("Danh mục yêu thích (khai báo lúc đăng ký)", cats, selection_mode="multi", key="fav",
                   format_func=lambda c: f"{ICON.get(c, '🛍️')} {c}", disabled=bool(ss.trail),
                   help="Đang có phiên xem nên hồ sơ dựng từ phiên; bấm “Xoá phiên” để dùng danh mục." if ss.trail else None)
    if ss.trail:
        groups, gcats, via = (tuple(ss.trail),), (None,), f"{len(ss.trail)} sản phẩm đã xem"
    else:
        groups = tuple(tuple(catalog[catalog["main_category"] == c]["n_events"].nlargest(3).index.tolist()) for c in fav)
        gcats, via = tuple(fav), f"bán chạy của {len(fav)} danh mục, xen kẽ"
    holder = st.empty()
    if not any(groups):
        with holder.container():
            explain([("Người dùng", "Thành viên mới · 0 lịch sử"), ("Định tuyến", "Chưa có hồ sơ"),
                     ("Mô-đun", "Thịnh hành 7 ngày"), ("Ứng viên", "Chọn danh mục hoặc xem sản phẩm để cá nhân hoá")], None)
        trending_shelf(k, "Chọn danh mục hoặc xem vài sản phẩm để cá nhân hoá.")
        return
    recs, ms = new_member_recs(groups, gcats, k)
    seeds = [p for g in groups for p in g]
    section("Gợi ý cho bạn", "Content-based", f"Hồ sơ dựng từ {via}; điểm cộng cho sản phẩm đang phổ biến.")
    items = []
    for p in recs:
        badges, note = match_badges(p, seeds)
        items.append({"pid": p, "badges": badges or [("Phổ biến", "orange")], "note": note})
    shelf(items, "n")
    with holder.container():
        explain([("Người dùng", "Thành viên mới · 0 lịch sử mua"),
                 ("Định tuyến", "Không đủ lịch sử cho ALS → content-based"),
                 ("Mô-đun", "Cosine(danh mục · thương hiệu · giá) + β·phổ biến"),
                 ("Ứng viên", f"Hồ sơ: {via} · {Counter(meta(p)['cat'] or 'chưa rõ' for p in recs).most_common(1)[0][0]} nhiều nhất")], ms)
    if ss.trail:
        session_shelf(k, engine, f"Vì bạn vừa xem {meta(ss.trail[-1])['title']}")


def member_home(k: int, engine: str):
    router = A["router"]
    heavy = list(dict.fromkeys(u for u in EX_USERS + A["demo_users"] if u in A["demo_users"]))[:40]
    light = light_users_by_info()[:20]
    kind = {**{u: "Nhiều lịch sử" for u in heavy}, **{u: "Ít lịch sử" for u in light}, UNSEEN_ID: "Chưa có lịch sử"}
    with st.container(horizontal=True, vertical_alignment="bottom", gap="medium"):
        uid = st.selectbox("Đăng nhập với tài khoản", list(kind), key="uid", width=360,
                           format_func=lambda u: f"{kind[u]} · {'tài khoản mới' if u == UNSEEN_ID else f'#{u}'}")
        hide_bought = st.toggle("Ẩn sản phẩm đã mua", value=True, key="hide_bought")
    group = router.group_of([uid])[uid]
    n_hist = int(router.counts.get(uid, 0))
    hist = A["history"][A["history"]["user_id"] == uid].sort_values("event_time")
    status: dict[int, int] = {}
    for p, e in zip(hist["product_id"], hist["event_type"]):
        status[p] = max(status.get(p, 0), RANK[e])
    if len(hist):
        tops = Counter(meta(p)["leaf"] or meta(p)["brand"] or "chưa rõ" for p in hist["product_id"].unique())
        with st.expander(f"Lịch sử · {len(hist)} sự kiện · hay quan tâm: " + ", ".join(n for n, _ in tops.most_common(3))):
            st.markdown(" ".join(f":{EVENT[e][1]}-badge[{EVENT[e][0]} · {meta(p)['title']}]"
                                 for p, e in list(zip(hist["product_id"], hist["event_type"]))[-12:]))
    rule = {"unseen": "0 tương tác", "light": f"1–{router.heavy_min - 1} tương tác",
            "heavy": f"≥ {router.heavy_min} tương tác"}[group]
    holder = st.empty()
    if group == "unseen":
        trending_shelf(k, "Tài khoản chưa có lịch sử nên bộ định tuyến chỉ có thể dùng độ phổ biến.")
        steps, ms = [("Ứng viên", f"{k} sản phẩm phổ biến nhất")], None
    else:
        recs, ranks, ms = member_recs(uid, k + 15)
        if hide_bought:
            recs = [p for p in recs if status.get(p, 0) < 3]
        recs = recs[:k]
        items, n_new = [], 0
        for p in recs:
            s = status.get(p, 0)
            badges = [(("Đã mua", "green") if s == 3 else ("Đã thêm giỏ", "orange") if s == 2
                       else ("Đã xem", "gray") if s == 1 else ("Mới", "blue"))]
            badges += [(f"{name} #{r[p]}", "violet") for name, r in ranks.items() if p in r]
            n_new += s == 0
            items.append({"pid": p, "badges": badges})
        section("Dành riêng cho bạn", module_name(A["routing"][group]),
                f"{n_new}/{len(recs)} là sản phẩm mới với người dùng này; phần còn lại họ đã tương tác "
                "(mô hình được thưởng khi gợi ý lại, xem mục Giới hạn). Nhãn tím = thứ hạng trong từng mô hình thành phần.")
        shelf(items, "m")
        both = sum(all(p in r for r in ranks.values()) for p in recs) if ranks else 0
        parts = [f"{name}: {sum(p in r for p in recs)}" for name, r in ranks.items()]
        steps = [("Ứng viên", " · ".join(parts + [f"cả hai: {both}"]) if ranks else f"{len(recs)} sản phẩm")]
    with holder.container():
        explain([("Người dùng", f"#{uid} · {n_hist} tương tác" if uid != UNSEEN_ID else "Tài khoản mới · 0 tương tác"),
                 ("Định tuyến", f"{rule} → nhóm {group}"),
                 ("Mô-đun", module_name(A["routing"][group]))] + steps, ms)
    if ss.trail:
        session_shelf(k, engine, f"Tiếp tục phiên · vì bạn vừa xem {meta(ss.trail[-1])['title']}")


@st.fragment
def cart_panel(k: int):
    """Own fragment: removing an item or adding a suggestion only re-runs this panel."""
    show_flash()
    cart = ss.cart
    with st.container(border=True):
        total = sum(meta(p)["price"] for p in cart)
        st.markdown(f"**🛒 Giỏ hàng** · {len(cart)} sản phẩm" + (f" · **{money(total)}**" if cart else ""))
        for p in cart:
            mini_row(p, [], "cart", action="remove")
        with st.container(horizontal=True, gap="small"):
            st.button("Giỏ mẫu", icon=":material/auto_awesome:", type="tertiary", on_click=set_state,
                      args=("cart", list(EX_CART)))
            st.button("Xoá giỏ", icon=":material/delete:", type="tertiary", on_click=set_state, args=("cart", []),
                      disabled=not cart)
        if not cart:
            st.caption("Giỏ trống. Thêm sản phẩm bằng nút 🛒 trên thẻ, hoặc bấm **Giỏ mẫu**.")
            return
        st.markdown("**Mua kèm** :gray-badge[FP-Growth]")
        rules, filler = fbt_split(cart, k)
        if rules:
            for p, conf in rules[:5]:
                mini_row(p, [(f"{conf:.0%} khách mua giỏ này cũng mua", "green")], "cr", action="cart")
        else:
            st.caption("Chưa có luật mua kèm đủ mạnh cho giỏ này; sản phẩm có ★ trong ô tìm kiếm có luật.")
        if filler:
            with st.expander(f"Bổ sung (không phải luật) · {len(filler)}"):
                for p, r in filler[:5]:
                    mini_row(p, [(r, "gray")], "cf", action="cart")


@st.dialog("Chi tiết sản phẩm", width="large", on_dismiss=close_qv)
def quick_view(k: int):
    pid = ss.qv  # read from state: buttons inside re-run only the dialog, which may switch product
    show_flash()
    m = meta(pid)
    c0, c1, c2 = st.columns([1, 4, 2], vertical_alignment="center")
    c0.html(tile(m, 88))
    c1.markdown(f"### {m['title']}")
    c1.html(badge_html([(m["leaf"] or "Chưa rõ danh mục", "gray")] + ([("★ Có luật mua cùng", "green")] if pid in HAS_RULE else []))
            + f"<div style='font-size:12px;opacity:.65'>#{pid} · {popularity(pid)}</div>")
    c2.markdown(f"## {money(m['price'])}")
    in_cart = pid in ss.cart
    c2.button("Đã có trong giỏ" if in_cart else "Thêm vào giỏ", icon=":material/add_shopping_cart:", type="primary",
              disabled=in_cart, on_click=add_to_cart, args=(pid,), key=f"qv_add_{pid}")
    st.caption(f"Đã ghi vào phiên (lượt xem thứ {len(ss.trail)}). Đóng hộp thoại để xem gợi ý cập nhật.")
    left, right = st.columns(2)
    with left:
        st.markdown("**Sản phẩm tương tự** :gray-badge[Item2Vec]")
        st.caption("Cùng ngữ cảnh xem trong phiên: thường là sản phẩm thay thế.")
        for p, s, _ in A["item2vec"].recommend_session([pid], 6):
            mini_row(p, [(f"Tương đồng {s:.2f}", "blue")], f"qi{pid}", action="view")
    with right:
        st.markdown("**Thường được mua cùng** :gray-badge[FP-Growth]")
        rules, _ = fbt_split([pid], k)
        if rules:
            st.caption("Chỉ gồm luật khai phá từ các giỏ hàng thật.")
            for p, conf in rules[:6]:
                mini_row(p, [(f"{conf:.0%} khách mua sản phẩm này cũng mua", "green")], f"qf{pid}", action="cart")
        else:
            st.caption("Sản phẩm này chưa có luật mua cùng đủ mạnh.")


# ---------------------------------------------------------------- pages
def shop():
    show_flash()
    with st.sidebar:
        st.header("Cài đặt")
        k = st.slider("Số gợi ý (K)", 5, 20, 10)
        engines = {"Bộ xếp hạng LightGBM (tốt nhất)": "ltr", "Lai: co-visitation → Item2Vec": "hybrid",
                   "Co-visitation": "covisit", "Item2Vec": "item2vec"}
        engines = {n: e for n, e in engines.items() if e in A}
        engine = engines[st.selectbox("Mô hình theo phiên", list(engines), help="Dùng cho gợi ý từ phiên đang xem.")]
        st.toggle("Hiện cách hệ thống quyết định", value=True, key="explain")
        show_unknown = st.checkbox("Tìm cả sản phẩm chưa phân loại", value=False,
                                   help="Khoảng 44% sản phẩm không có danh mục; ẩn khỏi ô tìm kiếm cho dễ đánh giá.")
        with st.expander("Giới hạn của demo"):
            st.markdown(
                "- Kết quả offline trên tuần 24–30/11/2019, mẫu 5% người dùng; không phải A/B test.\n"
                "- Nhãn đánh giá gồm cả tương tác lặp lại, nên mô hình có xu hướng gợi ý lại sản phẩm đã xem.\n"
                "- Dữ liệu không có tên/ảnh sản phẩm; tên hiển thị được sinh từ thương hiệu + danh mục.\n"
                "- 34% người dùng test chưa có lịch sử nên chỉ nhận gợi ý thịnh hành.\n"
                "- Gợi ý thiên về sản phẩm phổ biến (≈90% nằm trong top 1%, độ phủ catalog ≈2,6%).\n"
                "- “Mua kèm” gồm luật khai phá thật và phần bổ sung theo danh mục; hai phần được tách rõ.")

    with st.container(horizontal=True, vertical_alignment="center"):
        st.segmented_control("Bạn là ai?", PERSONAS, key="persona", label_visibility="collapsed", required=True)
        persona = ss.persona or GUEST
        st.space("stretch")
        if persona == GUEST:
            st.button("Đăng ký thành viên với phiên này", icon=":material/person_add:", type="primary", on_click=sign_up,
                      disabled=not ss.trail, help="Trở thành thành viên mới; hồ sơ được dựng từ các sản phẩm vừa xem.")
        else:
            st.button("Đăng xuất", icon=":material/logout:", on_click=sign_out, help="Quay về khách vãng lai, giữ nguyên phiên.")

    main, side = st.columns([3.3, 1], gap="medium")
    with main:
        trail_bar(build_picker(show_unknown))
        {GUEST: guest_home, NEW: new_member_home, MEMBER: member_home}[persona](k, engine)
    with side:
        cart_panel(k)

    if ss.qv is not None:
        quick_view(k)


def evaluation():
    st.title("📊 Đánh giá offline")
    if not SUMMARY:
        st.info("Chưa có `results/summary.json`. Chạy `python src/make_summary.py`.")
        return
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


st.navigation([st.Page(shop, title="Cửa hàng", icon="🛍️", url_path="shop", default=True),
               st.Page(evaluation, title="Đánh giá", icon="📊", url_path="danh-gia")], position="top").run()
