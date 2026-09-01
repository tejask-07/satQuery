from PIL import Image
from app.vlm.model import VLM

vlm = VLM()

img = Image.new('RGB', (10980, 10980), color = 'red')

images = {
    "before": img.copy(),
    "after": img.copy(),
    "change_map": img.copy(),
    "s1_composite": img.copy()
}

try:
    vlm.generate(question="Test question", evidence="Test evidence", images=images)
except Exception as e:
    print(e)
