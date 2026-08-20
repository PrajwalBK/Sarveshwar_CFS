import pytest
from app.ai.tier_estimator import TierEstimator
from app.ai.stacker_state_machine import ContainerTier


def test_tier_spreader_reference():
    estimator = TierEstimator({
        "tier_calibration": {
            "method": "spreader_reference",
            "spreader_real_height_m": 0.5,
            "container_real_height_m": 2.5
        }
    })

    # Spreader height = 50px -> 100px per meter (0.01 m/px)
    spreader_bbox = [500, 100, 700, 150]
    ground_y = 700.0

    # 1. Container at ground level (ground_gap = 0px) -> Tier 1
    container_tier1 = [500, 450, 700, 700]
    res1 = estimator.estimate(container_tier1, spreader_bbox, ground_y)
    assert res1 == ContainerTier.TIER_1

    # 2. Container elevated by 260px (~2.6 meters) -> Tier 2
    container_tier2 = [500, 190, 700, 440]
    res2 = estimator.estimate(container_tier2, spreader_bbox, ground_y)
    assert res2 == ContainerTier.TIER_2

    # 3. Container elevated by 520px (~5.2 meters) -> Tier 3
    container_tier3 = [500, 0, 700, 180]
    res3 = estimator.estimate(container_tier3, spreader_bbox, ground_y)
    assert res3 == ContainerTier.TIER_3


def test_tier_cabin_view():
    estimator = TierEstimator({
        "tier_calibration": {
            "method": "cabin_view",
            "tier_thresholds_px": [550, 420, 300, 180, 100]
        }
    })

    # Cabin angle: as container lifts up, container_bottom_y decreases
    assert estimator.estimate([300, 400, 900, 600]) == ContainerTier.TIER_1  # y2=600 >= 550
    assert estimator.estimate([300, 250, 900, 480]) == ContainerTier.TIER_2  # y2=480 >= 420
    assert estimator.estimate([300, 150, 900, 350]) == ContainerTier.TIER_3  # y2=350 >= 300
    assert estimator.estimate([300, 50,  900, 220]) == ContainerTier.TIER_4  # y2=220 >= 180
    assert estimator.estimate([300, 10,  900, 120]) == ContainerTier.TIER_5  # y2=120 >= 100


def test_tier_pixel_height():
    estimator = TierEstimator({
        "tier_calibration": {
            "method": "pixel_height",
            "tier_thresholds_px": [600, 450, 300, 200, 120]
        }
    })

    assert estimator.estimate([0, 0, 100, 650]) == ContainerTier.TIER_1
    assert estimator.estimate([0, 0, 100, 500]) == ContainerTier.TIER_2
    assert estimator.estimate([0, 0, 100, 350]) == ContainerTier.TIER_3
    assert estimator.estimate([0, 0, 100, 220]) == ContainerTier.TIER_4
    assert estimator.estimate([0, 0, 100, 150]) == ContainerTier.TIER_5
    assert estimator.estimate([0, 0, 100, 80]) == ContainerTier.TIER_5
