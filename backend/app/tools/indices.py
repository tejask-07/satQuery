def calculate_ndvi(**kwargs):
    return {
        "status": "success",
        "index": "NDVI",
        "mean_before": 0.71,
        "mean_after": 0.54,
        "change": -0.17,
    }


def calculate_ndwi(**kwargs):
    return {
        "status": "success",
        "index": "NDWI",
        "mean_before": 0.21,
        "mean_after": 0.48,
        "change": 0.27,
    }


def calculate_ndbi(**kwargs):
    return {
        "status": "success",
        "index": "NDBI",
        "mean_before": 0.18,
        "mean_after": 0.32,
        "change": 0.14,
    }