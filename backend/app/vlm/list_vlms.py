import os
import requests


URL = "https://router.huggingface.co/v1/models"


def main():
    token = os.getenv("HF_TOKEN")

    if not token:
        raise RuntimeError(
            "HF_TOKEN is not set in this PowerShell session."
        )

    response = requests.get(
        URL,
        headers={
            "Authorization": f"Bearer {token}"
        },
        timeout=30,
    )

    print("HTTP STATUS:", response.status_code)

    response.raise_for_status()

    data = response.json()

    models = data.get("data", [])

    print("\nTOTAL MODELS:", len(models))
    print("\nAVAILABLE MODELS:\n")

    for model in models:
        model_id = model.get("id", "unknown")

        print(model_id)


if __name__ == "__main__":
    main()