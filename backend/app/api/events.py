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


@router.post('/gate-events/{event_id}/sync')
def sync_event(event_id: str, request: Request):
    """Manually dispatch and synchronize a saved gate event to the CFS Smart Yard cloud API."""
    result = request.app.state.repository.event(event_id)
    if result is None:
        raise HTTPException(404, 'Event not found')
    dispatcher = getattr(getattr(request.app.state, 'runtime', None), 'dispatcher', None)
    if not dispatcher or not dispatcher.enabled:
        raise HTTPException(400, 'Prosper cloud sync is not enabled in backend configuration')
    success, resp = dispatcher.dispatch_event_record(result, request.app.state.snapshots)
    if not success:
        raise HTTPException(502, f"Failed to sync event to CFS Cloud: {resp.get('error', 'Unknown error')}")
    return {'status': 'synced', 'event_id': event_id, 'cloud_response': resp}




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
