"""Smoke test of the Streamlit demo: every user type and the cart flow render without exceptions.
Skipped when the artifacts have not been built (src/build_artifacts.py) or results/summary.json is missing."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (ROOT / "artifacts" / "models.pkl").exists() or not (ROOT / "results" / "summary.json").exists(),
                                reason="demo artifacts not built")


def test_demo_flows():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300).run()
    assert not at.exception
    for t in ("Thành viên mới", "Thành viên cũ"):
        at.segmented_control(key="user_type").set_value(t).run()
        assert not at.exception, t
    for i in (0, len(at.selectbox[0].options) - 1):  # a heavy user and the 'no history' user
        at.selectbox[0].select_index(i).run()
        assert not at.exception
    at.segmented_control(key="user_type").set_value("Khách vãng lai").run()
    [b for b in at.button if b.label == "Giỏ mẫu"][0].click().run()
    assert not at.exception and len(at.session_state.cart) == 2
    assert [t.label for t in at.tabs] == ["🏠 Trang chủ", "📦 Trang sản phẩm", "🛒 Giỏ hàng", "📊 Đánh giá"]
