import pytest
import imaging.thumb_atlas_pack as mod

CASES = [
    ("padding_bleed", (1)),
    ("mip_isolate", (1)),
    ("uv_cache", (1)),
    ("format_group", (1)),
    ("layout_determinism", (1)),
    ("size_align", (1)),
    ("pack", (1)),
    ("validation", (1)),
    ("audit", (1)),
    ("fallback", (1)),
    ("alpha_bucket", (1)),
    ("metrics", (1)),
    ("rebuild", (1)),
    ("sampling", (1)),
    ("determinism", (1)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
