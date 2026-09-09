from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

router = APIRouter(prefix='/api/cameras', tags=['cameras'])


def worker(request, camera_id):
    result = request.app.state.runtime.manager.workers.get(camera_id)
    if result is None:
        raise HTTPException(404, 'Camera not found')
    return result


@router.get('')
def cameras(request: Request):
    return request.app.state.runtime.camera_status()


@router.get('/sources/available')
def sources(request: Request):
    return request.app.state.runtime.available_camera_sources()


class SourceSelection(BaseModel):
    source_id: str


@router.post('/{camera_id}/source')
def select_source(camera_id: str, body: SourceSelection, request: Request):
    worker(request, camera_id)
    try:
        request.app.state.runtime.select_camera_source(camera_id, body.source_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from None
    return {'camera_id': camera_id, 'status': worker(request, camera_id).status()}


@router.get('/{camera_id}/status')
def status(camera_id: str, request: Request):
    return worker(request, camera_id).status()


@router.post('/{camera_id}/test')
def test_camera(camera_id: str, request: Request):
    return worker(request, camera_id).test()


@router.get('/{camera_id}/frame')
def preview(camera_id: str, request: Request):
    frame = worker(request, camera_id).latest()
    if frame is None:
        raise HTTPException(404, 'No frame available')
    import cv2
    image = frame.image
    if image.shape[1] > 640:
        image = cv2.resize(image, (640, round(image.shape[0] * 640 / image.shape[1])))
    ok, jpeg = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 75])
    if not ok:
        raise HTTPException(503, 'Preview unavailable')
    return Response(jpeg.tobytes(), media_type='image/jpeg', headers={'Cache-Control': 'no-store'})


@router.get('/{camera_id}/stream')
def stream_camera(camera_id: str, request: Request):
    cam = worker(request, camera_id)
    runtime = request.app.state.runtime

    def frame_generator():
        import cv2, time
        while True:
            frame = cam.latest()
            if frame is not None and frame.image is not None:
                img = frame.image.copy()
                try:
                    dets = runtime.recent_detections(limit=10, camera_id=camera_id)
                    now_ts = frame.timestamp
                    for d in dets:
                        d_ts_raw = d.get('frame_timestamp')
                        if d_ts_raw:
                            from datetime import datetime
                            d_ts = datetime.fromisoformat(d_ts_raw.rstrip('Z'))
                            if abs((now_ts - d_ts).total_seconds()) <= 0.8:
                                bbox = d.get('bbox')
                                cls_name = d.get('class_name', '')
                                conf = d.get('confidence', 0.0)
                                if bbox and len(bbox) == 4:
                                    bx1, by1, bx2, by2 = map(int, bbox)
                                    color = (0, 215, 255) if cls_name == 'feet' else (40, 190, 80)
                                    cv2.rectangle(img, (bx1, by1), (bx2, by2), color, 2)
                                    lbl = f"{cls_name.upper()} ({int(conf * 100)}%)"
                                    cv2.putText(img, lbl, (max(5, bx1), max(20, by1 - 6)),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                except Exception:
                    pass

                if img.shape[1] > 640:
                    img = cv2.resize(img, (640, round(img.shape[0] * 640 / img.shape[1])))
                ok, jpeg = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ok:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            time.sleep(0.04)  # ~25 FPS smooth playback

    return StreamingResponse(frame_generator(), media_type='multipart/x-mixed-replace; boundary=frame')


