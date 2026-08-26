def search_imagery(**kwargs):
    return {
        "status": "success",
        "source": "MOCK_SENTINEL_2",
        "images": [
            {
                "id": "mock_image_2021",
                "date": "2021-06-15",
                "cloud_cover": 8.2,
            },
            {
                "id": "mock_image_2025",
                "date": "2025-06-18",
                "cloud_cover": 6.7,
            },
        ],
    }