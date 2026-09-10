from typing import Protocol, Any
import logging
from app.domain import OCRRead

log = logging.getLogger('gate')


class OCREngine(Protocol):
    def read(self, image: Any) -> OCRRead: ...


class EasyOCREngine:
    def __init__(self, settings):
        import easyocr
        self.reader = easyocr.Reader(['en'], gpu=settings.ocr_gpu,
                                     model_storage_directory=str(settings.ocr_model_directory),
                                     download_enabled=settings.ocr_download_enabled, verbose=False)

    def read(self, image) -> OCRRead:
        import cv2
        if image is None or getattr(image, 'size', 0) == 0:
            return OCRRead('', 0.0)

        h, w = image.shape[:2]
        is_vertical = h > 1.5 * w

        proc_img = image
        if is_vertical and w < 100:
            scale = min(3.0, 120.0 / max(1, w))
            proc_img = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

        items = self.reader.readtext(
            proc_img, detail=1, paragraph=False,
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            rotation_info=[90, 180, 270] if not is_vertical else None,
            low_text=0.3, text_threshold=0.4, link_threshold=0.2
        )

        if not items and is_vertical:
            rot_cw = cv2.rotate(proc_img, cv2.ROTATE_90_CLOCKWISE)
            items = self.reader.readtext(
                rot_cw, detail=1, paragraph=False,
                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            )

        if not items:
            return OCRRead('', 0.0)

        if is_vertical:
            items = sorted(items, key=lambda it: it[0][0][1])
        else:
            items = sorted(items, key=lambda it: (it[0][0][1] // 30, it[0][0][0]))

        combined_text = ' '.join(str(item[1]) for item in items)[:4096]
        from app.ocr.validator import correct_feet_size_codes
        combined_text = correct_feet_size_codes(combined_text)
        min_conf = float(min(item[2] for item in items))
        return OCRRead(combined_text, min_conf)


class OlmOCREngine:
    """
    oLmOCR / Qwen2-VL Vision-Language OCR Engine.
    Specialized for container code extraction across complex orientations
    (vertical text, stacked characters, low-contrast stamps).
    """
    def __init__(self, settings):
        import torch
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        self.model_name = getattr(settings, 'ocr_model_name', 'Qwen/Qwen2-VL-2B-Instruct')
        self.device = 'cuda' if (getattr(settings, 'ocr_gpu', True) and torch.cuda.is_available()) else 'cpu'
        self.torch_dtype = torch.float16 if self.device == 'cuda' else torch.float32

        log.info(f'Loading oLmOCR (Qwen2-VL) model {self.model_name} on {self.device}...')
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=self.torch_dtype,
            device_map=self.device
        )
        self.prompt = (
            "You are a container OCR system. Read the container owner code and serial number printed vertically or horizontally on this container column (e.g. GXYU 509070 4). "
            "Output ONLY the letters and digits, separated by spaces. Do not output words or explanations."
        )
        log.info(f'oLmOCR model loaded successfully on {self.device}.')

    def read(self, image) -> OCRRead:
        import cv2
        import re
        import torch
        from PIL import Image
        from qwen_vl_utils import process_vision_info

        if image is None or getattr(image, 'size', 0) == 0:
            return OCRRead('', 0.0)

        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)

            messages = [
                {
                    'role': 'user',
                    'content': [
                        {'type': 'image', 'image': pil_image},
                        {'type': 'text', 'text': self.prompt}
                    ]
                }
            ]

            text_template = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text_template],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors='pt'
            ).to(self.device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=32,
                    do_sample=False
                )
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0].strip()

            if not output_text:
                return OCRRead('', 0.0)

            # Strip conversational text (e.g. "The container number is GXYU 509070 4")
            cleaned = output_text.replace('\n', ' ').replace('-', ' ')
            code_match = re.search(r'([A-Z]{3,4}\s*[0-9]{6,7})', cleaned.upper())
            if code_match:
                output_text = code_match.group(1).strip()
            else:
                output_text = cleaned.strip()

            from app.ocr.validator import correct_feet_size_codes
            output_text = correct_feet_size_codes(output_text)

            has_letters = any(c.isalpha() for c in output_text)
            has_digits = any(c.isdigit() for c in output_text)
            if has_letters and has_digits:
                confidence = 0.95
            elif has_digits:
                confidence = 0.85
            else:
                confidence = 0.70

            return OCRRead(output_text[:4096], confidence)

        except Exception as exc:
            log.error(f'oLmOCR read error: {exc}', exc_info=True)
            return OCRRead('', 0.0)


def create_ocr_engine(settings) -> OCREngine:
    engine_type = getattr(settings, 'ocr_engine', 'easyocr').lower()
    if engine_type in ('olmocr', 'qwen2_vl', 'qwen2vl', 'qwen'):
        return OlmOCREngine(settings)
    return EasyOCREngine(settings)
