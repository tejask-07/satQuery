from PIL import Image

from app.api import routes_query
from app.schemas.query import QueryRequest


class FakeQwen:
    def __init__(self):
        self.calls = []

    def get_model_info(self):
        return {
            "name": "Qwen/Qwen2.5-VL-3B-Instruct",
            "backend": "qwen",
            "status": "available",
            "adapter_loaded": False,
        }

    def answer(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "answer": "Visible features include supported optical scene content.",
            "status": "available",
            "adapter_loaded": False,
            "confidence": None,
        }

    def explain_change(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "answer": "Optical change analysis completed.",
            "status": "available",
            "adapter_loaded": False,
            "confidence": None,
        }


def _p2_response(task="single_image_vqa"):
    image = Image.new("RGB", (8, 8), color="green")
    return {
        "plan": {"task": task},
        "evidence_package": {"source": "test"},
        "images": {"before": image, "after": image},
    }


def test_missing_optional_s1_does_not_crash_generate_vlm_answer(monkeypatch):
    fake_qwen = FakeQwen()
    monkeypatch.setattr(routes_query, "load_p2_images", lambda evidence: _p2_response()["images"])
    monkeypatch.setattr(
        routes_query,
        "build_s1_visualization",
        lambda name: (_ for _ in ()).throw(FileNotFoundError("S1 patch is not available in the local cache.")),
    )
    monkeypatch.setattr(routes_query, "get_qwen_vlm", lambda: fake_qwen)

    result = routes_query.generate_vlm_answer(
        QueryRequest(query="What man-made features can you see?"),
        _p2_response(),
    )

    assert result["status"] == "available"
    assert result["model"]["name"] == "Qwen/Qwen2.5-VL-3B-Instruct"
    assert result["model"]["adapter_loaded"] is False
    assert result["model"]["s1_visualization_available"] is False
    assert "S1 patch is not available" in result["model"]["s1_visualization_error"]
    assert fake_qwen.calls[0]["image"] is not None


def test_available_s1_continues_existing_vlm_path(monkeypatch):
    fake_qwen = FakeQwen()
    s1_image = Image.new("RGB", (8, 8), color="blue")
    monkeypatch.setattr(routes_query, "load_p2_images", lambda evidence: _p2_response()["images"])
    monkeypatch.setattr(routes_query, "build_s1_visualization", lambda name: s1_image)
    monkeypatch.setattr(routes_query, "get_qwen_vlm", lambda: fake_qwen)

    result = routes_query.generate_vlm_answer(
        QueryRequest(query="What man-made features can you see?"),
        _p2_response(),
    )

    assert result["status"] == "available"
    assert result["model"]["s1_visualization_available"] is True
    assert result["model"]["s1_visualization_error"] is None
    assert fake_qwen.calls[0]["image"] is not None
