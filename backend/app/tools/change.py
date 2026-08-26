def compare_images(**kwargs):
    return {
        "status": "success",
        "comparison": "completed",
        "changed_pixels": 18452,
    }


def detect_change(**kwargs):
    return {
        "status": "success",
        "regions_detected": 7,
        "affected_area_km2": 4.82,
        "change_type": "candidate_change",
    }