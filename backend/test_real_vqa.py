from PIL import Image
from app.vlm.vqa import run_vqa

image = Image.open("satellite.jpg")

question = "What is visible in this satellite image?"

result = run_vqa(
    image=image,
    question=question,
    modality="optical"
)

print("\nVQA RESULT:")
print(result)