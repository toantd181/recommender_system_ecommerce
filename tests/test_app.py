"""Smoke test of the Streamlit demo: browsing builds the session, sign-up carries it over, every user type,
the cart and the evaluation page render without exceptions.
Skipped when the artifacts have not been built (src/build_artifacts.py) or results/summary.json is missing."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (ROOT / "artifacts" / "models.pkl").exists() or not (ROOT / "results" / "summary.json").exists(),
                                reason="demo artifacts not built")


def _btn(at, label):
    return [b for b in at.button if b.label == label]


def test_demo_flows():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300).run()
    assert not at.exception
    n0 = len(at.session_state.trail)

    # browsing: 'Xem' on a card opens the quick view and extends the session
    _btn(at, "Xem")[0].click().run()
    assert not at.exception and len(at.session_state.trail) == n0 + 1 and at.session_state.qv is not None
    at.session_state.qv = None
    at.run()

    # empty session -> trending; sample session back
    _btn(at, "Xoá phiên")[0].click().run()
    assert not at.exception and at.session_state.trail == []
    _btn(at, "Phiên mẫu")[0].click().run()
    assert not at.exception and at.session_state.trail

    # sign up from the guest session: becomes a new member, profile from the trail
    _btn(at, "Đăng ký thành viên với phiên này")[0].click().run()
    assert not at.exception and at.session_state.persona == "Thành viên mới" and at.session_state.from_guest > 0

    # returning members: a heavy user and the 'no history' account
    at.segmented_control(key="persona").set_value("Thành viên cũ").run()
    assert not at.exception
    sb = [s for s in at.selectbox if s.key == "uid"][0]
    for i in (0, len(sb.options) - 1):
        [s for s in at.selectbox if s.key == "uid"][0].select_index(i).run()
        assert not at.exception

    # cart
    at.segmented_control(key="persona").set_value("Khách vãng lai").run()
    _btn(at, "Giỏ mẫu")[0].click().run()
    assert not at.exception and len(at.session_state.cart) == 2


def test_evaluation_page():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300).run()
    from streamlit.navigation.page import calc_hash
    # switch_page() only takes page files; pages here are functions, identified by the hash of url_path
    at._page_hash = calc_hash("danh-gia")
    at.run()
    assert not at.exception
    assert any("Đánh giá offline" in t.value for t in at.title)
