import pytest
import numpy as np
from app.ai.stacker_state_machine import StackerStateMachine, StackerState, FrameObservation


def test_initial_state():
    fsm = StackerStateMachine()
    assert fsm.current_state == StackerState.EMPTY_SPREADER
    assert not fsm.container_attached


def test_approaching_and_contact():
    fsm = StackerStateMachine({"state_machine": {"approach_distance_px": 200}})
    
    # 1. Spreader far from container -> EMPTY_SPREADER
    obs1 = FrameObservation(
        frame_count=1,
        timestamp=1.0,
        spreader_bbox=[100, 100, 200, 150],
        container_bbox=[500, 500, 700, 600]
    )
    state, attached = fsm.update(obs1)
    assert state == StackerState.EMPTY_SPREADER
    assert not attached

    # 2. Spreader approaches container -> APPROACHING_CONTAINER
    obs2 = FrameObservation(
        frame_count=2,
        timestamp=1.1,
        spreader_bbox=[480, 350, 680, 400],
        container_bbox=[500, 400, 700, 600]
    )
    state, attached = fsm.update(obs2)
    assert state == StackerState.APPROACHING_CONTAINER

    # 3. Spreader makes contact with top of container -> CONTACT
    obs3 = FrameObservation(
        frame_count=3,
        timestamp=1.2,
        spreader_bbox=[500, 380, 700, 420],
        container_bbox=[500, 400, 700, 600]
    )
    state, attached = fsm.update(obs3)
    assert state == StackerState.CONTACT


def test_container_attachment_conditions():
    fsm = StackerStateMachine({
        "state_machine": {
            "co_motion_min_frames": 3,
            "alignment_iou_thresh": 0.2,
            "ground_gap_min_px": 10
        }
    })

    # Simulate 5 consecutive frames of co-motion and increasing ground gap
    ground_y = 700.0
    for i in range(6):
        obs = FrameObservation(
            frame_count=i + 1,
            timestamp=1.0 + i * 0.1,
            spreader_bbox=[500, 380 - i * 10, 700, 420 - i * 10],
            container_bbox=[500, 400 - i * 10, 700, 600 - i * 10], # moving up
            ground_y=ground_y
        )
        state, attached = fsm.update(obs)

    assert attached
    assert state in [StackerState.CONTAINER_ATTACHED, StackerState.LIFTING]


def test_attachment_failure_on_missing_spreader():
    fsm = StackerStateMachine()
    obs = FrameObservation(
        frame_count=1,
        timestamp=1.0,
        spreader_bbox=None, # Missing spreader
        container_bbox=[500, 400, 700, 600]
    )
    state, attached = fsm.update(obs)
    assert not attached
