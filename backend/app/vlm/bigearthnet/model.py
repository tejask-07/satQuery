def generate(
    self,
    image: Image.Image,
    question: str,
) -> str:
    prompt = f"""
You are analyzing a BigEarthNet Sentinel-2 satellite image.

The image coordinate system is normalized:
- top-left = (0.0, 0.0)
- top-right = (1.0, 0.0)
- bottom-left = (0.0, 1.0)
- bottom-right = (1.0, 1.0)

When a question contains:
<point>(x, y)</point>

the point is a normalized coordinate INSIDE the image,
where x is the horizontal position and y is the vertical
position.

For a bounding-box question:
1. Locate the land-cover region/object associated with
   the given point.
2. Find the boundaries of that region in the image.
3. Return its bounding box using normalized coordinates:
   [x1, y1, x2, y2]
4. x1/y1 = top-left corner.
5. x2/y2 = bottom-right corner.
6. All values must be between 0 and 1.

IMPORTANT OUTPUT RULE:
For a bounding-box question, output ONLY:
[x1, y1, x2, y2]

Do not explain your reasoning.
Do not say the point is outside the image unless the
coordinate is actually outside the [0,1] × [0,1] range.

Question:
{question}
"""

    image_url = self.image_to_data_url(image)


    response = self.client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url,
                        },
                    },
                ],
            }
        ],
        max_tokens=32,
    )

    return response.choices[0].message.content