from typing import Literal
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse

router = APIRouter(prefix='/api', tags=['events'])


@router.get('/gate-events')
def events(request: Request, limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0),
           container: str | None = Query(None, max_length=11), camera_id: str | None = None,
           event_type: Literal['ENTRY', 'EXIT', 'UNKNOWN'] | None = None,
           status: Literal['CONFIRMED', 'NEEDS_REVIEW'] | None = None):
    return request.app.state.repository.events(limit, offset, container, camera_id, event_type, status)


@router.get('/gate-events/{event_id}')
def event(event_id: str, request: Request):
    result = request.app.state.repository.event(event_id)
    if result is None:
        raise HTTPException(404, 'Event not found')
    return result


@router.get('/gate-events/{event_id}/json')
def event_json(event_id: str, request: Request):
    result = request.app.state.repository.event(event_id)
    if result is None:
        raise HTTPException(404, 'Event not found')
    json_path = request.app.state.snapshots.root / event_id / 'event.json'
    if not json_path.is_file():
        json_path = request.app.state.snapshots.save_event_json(event_id, result)
    return FileResponse(json_path, media_type='application/json', filename=f'event_{event_id}.json')



@router.get('/ocr-results')
def ocr_results(request: Request, limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0)):
    return request.app.state.repository.ocr_results(limit, offset)


@router.get('/snapshots/{snapshot_id}')
def snapshot(snapshot_id: str, request: Request):
    row = request.app.state.repository.snapshot(snapshot_id)
    if row is None:
        raise HTTPException(404, 'Snapshot not found')
    path = request.app.state.snapshots.path(row['image_path'])
    if not path.is_file():
        raise HTTPException(404, 'Snapshot file unavailable')
    return FileResponse(path, media_type='image/jpeg')
