"""
app/routes/scan.py — POST /scan, POST /scan/manual, GET /demo/{case_id}

This is the main scan pipeline orchestrator.
"""

from __future__ import annotations
import uuid
import json
import logging
from datetime import datetime, timezone
from typing import Optional
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Scan
from app.schemas import (
    ScanResult, ManualScanRequest,
    QualityResult, OcrResult, OcrField, CodeResult,
    CrossCheckItem, VisualResult, AnomalyRegion,
    AuthenticityResult, RegistryStatus, AlertMatch, CloneResult as CloneSchema,
    Reason, ScoreBreakdown, FieldRow,
)
from app.config import DEMO_CASES_DIR, OCR_CONFIDENCE_MIN
from app.pipeline.quality import assess_quality
from app.pipeline.ocr import run_ocr
from app.pipeline.fields import extract_fields, ExtractedFields
from app.pipeline.codes import decode_codes, DecodeResult
from app.pipeline.visual import run_visual
from app.pipeline.authenticity import (
    match_alert, registry_lookup, detect_clone, record_scan_event,
    hash_field, normalise_batch,
)
from app.pipeline.rules import evaluate_hard_rules
from app.pipeline.fusion import fuse
from app.pipeline.explain import localise_reasons, localise_next_steps

logger = logging.getLogger(__name__)
router = APIRouter()


def _ts_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _round2(v: Optional[float]) -> Optional[float]:
    return round(v, 2) if v is not None else None


async def _run_full_pipeline(
    img_bytes: bytes,
    img2_bytes: Optional[bytes],
    device_token: str,
    lat: Optional[float],
    lng: Optional[float],
    language: str,
    db: Session,
    extra_simulated_flags: Optional[list[str]] = None,
) -> ScanResult:
    """Core pipeline: quality → OCR → codes → visual → authenticity → rules → fusion."""

    scan_id = str(uuid.uuid4())
    ts = _ts_now()
    simulated_flags: list[str] = list(extra_simulated_flags or [])

    # ── 1. Quality ────────────────────────────────────────────────────────
    quality_report = assess_quality(img_bytes)
    quality_schema = QualityResult(
        ok=quality_report.ok,
        blur=quality_report.blur,
        brightness=quality_report.brightness,
        glare=quality_report.glare,
        contrast=quality_report.contrast,
        resolution_min=quality_report.resolution_min,
        tips=quality_report.tips,
    )

    if not quality_report.ok:
        result = ScanResult(
            id=scan_id,
            ts=ts,
            label="insufficient_quality",
            partial=False,
            quality=quality_schema,
            ocr=OcrResult(ran=False),
            code=CodeResult(ran=False),
            visual=VisualResult(ran=False),
            authenticity=AuthenticityResult(ran=False),
            not_checked=[
                "OCR text extraction (image quality insufficient)",
                "2D code verification (image quality insufficient)",
                "Visual similarity check (image quality insufficient)",
            ],
            next_steps=localise_next_steps("insufficient_quality", language),
            simulated_flags=simulated_flags,
        )
        _store_scan(db, result, device_token)
        return result

    # ── 2. OCR ────────────────────────────────────────────────────────────
    merged_text, ocr_blocks = run_ocr(img_bytes)
    avg_conf = (
        sum(b.confidence for b in ocr_blocks) / len(ocr_blocks)
        if ocr_blocks else 0.5
    )
    fields = extract_fields(merged_text, avg_block_confidence=avg_conf)

    ocr_schema = OcrResult(
        ran=bool(merged_text),
        name=OcrField(value=fields.name.value, confidence=fields.name.confidence) if fields.name.value else None,
        strength=OcrField(value=fields.strength.value, confidence=fields.strength.confidence) if fields.strength.value else None,
        form=OcrField(value=fields.form.value, confidence=fields.form.confidence) if fields.form.value else None,
        batch=OcrField(value=fields.batch.value, confidence=fields.batch.confidence) if fields.batch.value else None,
        mfg_date=OcrField(value=fields.mfg_date.value, confidence=fields.mfg_date.confidence) if fields.mfg_date.value else None,
        expiry=OcrField(value=fields.expiry.value, confidence=fields.expiry.confidence) if fields.expiry.value else None,
        manufacturer=OcrField(value=fields.manufacturer.value, confidence=fields.manufacturer.confidence) if fields.manufacturer.value else None,
    )

    # ── 3. Codes ──────────────────────────────────────────────────────────
    code_decode = decode_codes(img_bytes)
    if img2_bytes and not code_decode.gtin:
        code_decode2 = decode_codes(img2_bytes)
        if code_decode2.ran and code_decode2.raw:
            code_decode = code_decode2

    code_schema = CodeResult(
        ran=code_decode.ran,
        type=code_decode.code_type,
        raw=code_decode.raw,
        gtin=code_decode.gtin,
        batch=code_decode.batch,
        expiry=code_decode.expiry,
        serial=code_decode.serial,
        has_serial=code_decode.has_serial,
        gtin_valid=code_decode.gtin_valid,
        payload_source=code_decode.payload_source,
        url_offline_note=code_decode.url_offline_note,
    )

    # ── 4. Visual ─────────────────────────────────────────────────────────
    med_id = fields.matched_medicine.get("id") if fields.matched_medicine else None
    visual_report = run_visual(img_bytes, medicine_id=med_id)
    visual_schema = VisualResult(
        ran=visual_report.ran,
        identity_score=visual_report.identity_score,
        alignment_ok=visual_report.alignment_ok,
        heatmap_url=visual_report.heatmap_url,
        anomaly_score=visual_report.anomaly_score,
        anomaly_regions=[
            AnomalyRegion(label=r.label, bbox=r.bbox, score=r.score)
            for r in (visual_report.anomaly_regions or [])
        ],
    )

    # ── 5. Authenticity ───────────────────────────────────────────────────
    # Privacy: round lat/lng
    lat_r = _round2(lat)
    lng_r = _round2(lng)

    # Hash serial and batch
    serial_hash = hash_field(code_decode.serial) if code_decode.serial else None
    batch_val = fields.batch.value or code_decode.batch
    batch_hash = hash_field(normalise_batch(batch_val)) if batch_val else None

    # Alert match
    brand_candidate = (
        fields.matched_medicine.get("name") if (fields and fields.matched_medicine) else None
    ) or (fields.name.value if fields else None)
    alert_result = match_alert(
        db,
        name=brand_candidate,
        batch=batch_val,
        expiry=fields.expiry.value or code_decode.expiry,
    )

    # Registry
    registry_result = registry_lookup(db, code_decode.gtin, code_decode.serial)
    if registry_result.simulated:
        simulated_flags.append("registry")

    # Clone
    clone_result = detect_clone(
        db,
        serial_hash=serial_hash,
        batch_hash=batch_hash,
        current_printed_batch=fields.batch.value,
        current_printed_expiry=fields.expiry.value,
        current_printed_name=fields.name.value,
        current_device_token=device_token,
        current_lat=lat_r,
        current_lng=lng_r,
        current_ts=datetime.now(timezone.utc),
        registry_status=registry_result.status,
    )

    auth_schema = AuthenticityResult(
        ran=True,
        registry=RegistryStatus(status=registry_result.status, simulated=registry_result.simulated),
        alert=AlertMatch(
            match_level=alert_result.match_level,
            alert_id=alert_result.alert_id,
            source_month=alert_result.source_month,
            source_url=alert_result.source_url,
            snapshot_date=alert_result.snapshot_date,
            snapshot_stale=alert_result.snapshot_stale,
        ) if alert_result else None,
        clone=CloneSchema(
            signal=clone_result.signal,
            verdict_text=clone_result.verdict_text,
        ) if clone_result else None,
    )

    # ── 6. Hard rules ──────────────────────────────────────────────────────
    hard_rules = evaluate_hard_rules(
        fields=fields,
        code=code_decode,
        alert=alert_result,
        clone=clone_result,
        registry=registry_result,
    )

    # ── 7. Fusion ──────────────────────────────────────────────────────────
    fusion = fuse(
        quality_ok=quality_report.ok,
        hard_rules=hard_rules,
        fields=fields,
        code=code_decode,
        visual=visual_report,
        alert=alert_result,
        clone=clone_result,
        registry=registry_result,
        partial=False,
    )

    # ── 8. Localise ────────────────────────────────────────────────────────
    reasons_local = localise_reasons(fusion.reasons, language)
    next_steps_local = localise_next_steps(fusion.label, language)

    # Cross-check schema
    cross_check_items = [
        CrossCheckItem(
            field=cc["field"],
            printed=cc.get("printed"),
            code=cc.get("code"),
            status=cc["status"],
            bbox=cc.get("bbox"),
        )
        for cc in fusion.cross_check
    ]

    field_rows_items = [
        FieldRow(
            field=fr["field"],
            printed=fr.get("printed") or (getattr(fields, fr["field"]).value if fields and hasattr(fields, fr["field"]) and getattr(fields, fr["field"]) else None),
            code_value=fr.get("code_value") or (getattr(code_decode, fr["field"]) if code_decode and hasattr(code_decode, fr["field"]) else None),
            expected=fr.get("expected") or (fields.matched_medicine.get("name") if fr["field"] == "name" and fields and fields.matched_medicine else None),
            status=fr["status"],
            bbox=fr.get("bbox"),
        )
        for fr in fusion.field_rows
    ]

    reasons_items = [
        Reason(
            id=r["id"],
            text=r["text"],
            severity=r["severity"],
            contribution=r.get("contribution", 0),
        )
        for r in reasons_local
    ]

    result = ScanResult(
        id=scan_id,
        ts=ts,
        label=fusion.label,
        partial=False,
        quality=quality_schema,
        ocr=ocr_schema,
        code=code_schema,
        cross_check=cross_check_items,
        visual=visual_schema,
        authenticity=auth_schema,
        reasons=reasons_items,
        score=fusion.score,
        score_breakdown=ScoreBreakdown(**fusion.score_breakdown),
        checked=fusion.checked,
        not_checked=fusion.not_checked,
        confidence=fusion.confidence,
        confidence_note=fusion.confidence_note,
        field_rows=field_rows_items,
        next_steps=next_steps_local,
        simulated_flags=simulated_flags,
        disclaimer="Screening only. This does not confirm authenticity.",
    )

    # ── 9. Store scan ──────────────────────────────────────────────────────
    _store_scan(db, result, device_token)

    # ── 10. Record scan event ─────────────────────────────────────────────
    if batch_hash or serial_hash:
        record_scan_event(
            db,
            serial_hash=serial_hash,
            batch_hash=batch_hash,
            district=_district_from_lat_lng(lat_r, lng_r),
            lat_round=lat_r,
            lng_round=lng_r,
            ts=datetime.now(timezone.utc),
            device_token=device_token,
            label=fusion.label,
            quality_ok=True,
            printed_batch=fields.batch.value,
            printed_expiry=fields.expiry.value,
            printed_name=fields.name.value,
        )

    return result


def _district_from_lat_lng(lat: Optional[float], lng: Optional[float]) -> Optional[str]:
    """Placeholder district lookup — returns None when no data available."""
    return None


def _store_scan(db: Session, result: ScanResult, device_token: str, session_id: Optional[str] = None) -> None:
    """Persist ScanResult to DB."""
    try:
        top_reason = result.reasons[0].text if result.reasons else None
        scan = Scan(
            id=result.id,
            device_token=device_token,
            ts=datetime.now(timezone.utc),
            label=result.label,
            partial=result.partial,
            name=result.ocr.name.value if result.ocr and result.ocr.name else None,
            batch=result.ocr.batch.value if result.ocr and result.ocr.batch else None,
            expiry=result.ocr.expiry.value if result.ocr and result.ocr.expiry else None,
            manufacturer=result.ocr.manufacturer.value if result.ocr and result.ocr.manufacturer else None,
            top_reason=top_reason,
            result_json=result.model_dump(),
            session_id=session_id or result.session_id,
            simulated=bool(result.simulated_flags),
        )
        db.add(scan)
        db.commit()
    except Exception as exc:
        logger.error("scan: store error: %s", exc)
        db.rollback()


# ── POST /scan ────────────────────────────────────────────────────────────

@router.post("/scan", response_model=ScanResult)
async def post_scan(
    image: UploadFile = File(...),
    image2: Optional[UploadFile] = File(None),
    device_token: str = Form(...),
    lat: Optional[float] = Form(None),
    lng: Optional[float] = Form(None),
    language: str = Form("en"),
    db: Session = Depends(get_db),
):
    """Full scan pipeline."""
    img_bytes = await image.read()
    img2_bytes = await image2.read() if image2 else None

    return await _run_full_pipeline(
        img_bytes=img_bytes,
        img2_bytes=img2_bytes,
        device_token=device_token,
        lat=lat,
        lng=lng,
        language=language,
        db=db,
    )


# ── POST /scan/manual ─────────────────────────────────────────────────────

@router.post("/scan/manual", response_model=ScanResult)
async def post_scan_manual(
    req: ManualScanRequest,
    db: Session = Depends(get_db),
):
    """Manual field-entry scan — no image, no visual. Always partial. Never low_risk."""
    scan_id = str(uuid.uuid4())
    ts = _ts_now()
    simulated_flags: list[str] = []
    language = req.language or "en"

    # Build fields from request
    from app.pipeline.fields import ExtractedFields, FieldResult
    fields = ExtractedFields()
    if req.name:
        fields.name = FieldResult(value=req.name, confidence=0.95)
    if req.batch:
        fields.batch = FieldResult(value=req.batch, confidence=0.95)
    if req.expiry:
        fields.expiry = FieldResult(value=req.expiry, confidence=0.95)
    if req.mfg_date:
        fields.mfg_date = FieldResult(value=req.mfg_date, confidence=0.90)
    if req.manufacturer:
        fields.manufacturer = FieldResult(value=req.manufacturer, confidence=0.90)
    if req.strength:
        fields.strength = FieldResult(value=req.strength, confidence=0.90)
    if req.form:
        fields.form = FieldResult(value=req.form, confidence=0.90)

    # Validate expiry
    if req.expiry:
        from app.pipeline.fields import _validate_expiry
        _validate_expiry(fields)

    # Fuzzy name match
    if req.name:
        from app.pipeline.fields import _fuzzy_name_match
        matched, flag, _ = _fuzzy_name_match(req.name)
        fields.name_flag = flag
        fields.matched_medicine = matched

    # Check if a barcode was provided directly or via batch field
    from app.pipeline.codes import _classify_and_parse, _looks_like_gs1
    raw_code = (req.barcode or "").strip()
    if not raw_code and req.batch:
        b_str = req.batch.strip()
        if _looks_like_gs1(b_str) or b_str.upper().startswith("MEDSURE|") or b_str.startswith("http"):
            raw_code = b_str

    if raw_code:
        code_decode = _classify_and_parse(raw_code)
    else:
        code_decode = DecodeResult(ran=False)

    if req.gtin and not code_decode.gtin:
        code_decode.gtin = req.gtin
    if req.serial and not code_decode.serial:
        code_decode.serial = req.serial
        code_decode.has_serial = bool(code_decode.serial)

    # If code was decoded, populate fields if not already provided
    if code_decode.ran:
        if code_decode.batch and (not fields.batch.value or fields.batch.value == raw_code):
            fields.batch = FieldResult(value=code_decode.batch, confidence=0.99)
        if code_decode.expiry and not fields.expiry.value:
            fields.expiry = FieldResult(value=code_decode.expiry, confidence=0.99)
            from app.pipeline.fields import _validate_expiry
            _validate_expiry(fields)
        if code_decode.gtin and not fields.matched_medicine:
            from app.config import MEDICINES_JSON
            import json as _json
            try:
                meds = _json.load(open(MEDICINES_JSON, encoding="utf-8"))
                for m in meds:
                    if m.get("gtin") == code_decode.gtin:
                        fields.matched_medicine = m
                        if not fields.name.value:
                            fields.name = FieldResult(value=m["name"], confidence=0.95)
                        break
            except Exception:
                pass

    # Alert match
    batch_val = fields.batch.value or req.batch
    alert_result = match_alert(
        db,
        name=fields.name.value or req.name,
        batch=batch_val,
        expiry=fields.expiry.value or req.expiry,
    )

    # Registry lookup
    registry_result = registry_lookup(db, code_decode.gtin, code_decode.serial)
    if registry_result.simulated:
        simulated_flags.append("registry")

    # Clone check
    clone_result = None
    if code_decode.has_serial or fields.batch.value or code_decode.batch:
        serial_hash = hash_field(code_decode.serial) if code_decode.serial else None
        b_val = fields.batch.value or code_decode.batch
        batch_hash = hash_field(normalise_batch(b_val)) if b_val else None
        clone_result = detect_clone(
            db,
            serial_hash=serial_hash,
            batch_hash=batch_hash,
            current_printed_batch=fields.batch.value,
            current_printed_expiry=fields.expiry.value,
            current_printed_name=fields.name.value,
            current_device_token=req.device_token,
            current_lat=None,
            current_lng=None,
            current_ts=datetime.now(timezone.utc),
            registry_status=registry_result.status,
        )

    visual_report_dummy = type("V", (), {"ran": False, "not_checked_reason": "Manual entry — no image", "identity_score": None, "alignment_ok": None, "heatmap_url": None, "anomaly_score": None, "anomaly_regions": []})()

    hard_rules = evaluate_hard_rules(
        fields=fields,
        code=code_decode,
        alert=alert_result,
        clone=clone_result,
        registry=registry_result,
    )

    fusion = fuse(
        quality_ok=True,  # no quality check for manual
        hard_rules=hard_rules,
        fields=fields,
        code=code_decode,
        visual=None,
        alert=alert_result,
        clone=clone_result,
        registry=registry_result,
        partial=True,
    )

    # Manual NEVER returns low_risk
    if fusion.label == "low_risk":
        fusion.label = "needs_verification"

    reasons_local = localise_reasons(fusion.reasons, language)
    next_steps_local = localise_next_steps(fusion.label, language)

    # Build CodeResult schema
    code_schema = CodeResult(
        ran=code_decode.ran,
        type=code_decode.code_type,
        raw=code_decode.raw,
        gtin=code_decode.gtin,
        batch=code_decode.batch,
        expiry=code_decode.expiry,
        serial=code_decode.serial,
        has_serial=code_decode.has_serial,
        gtin_valid=code_decode.gtin_valid,
        payload_source=code_decode.payload_source,
        url_offline_note=code_decode.url_offline_note,
    )

    # Determine printed vs code values
    printed_name = req.name or (fields.name.value if fields else None)
    printed_batch = req.batch if (req.batch and req.batch != raw_code) else (fields.batch.value if fields else None)
    printed_expiry = req.expiry if (req.expiry and req.expiry != raw_code) else (fields.expiry.value if fields else None)

    code_batch = code_decode.batch if code_decode.ran else None
    code_expiry = code_decode.expiry if code_decode.ran else None

    if raw_code and not req.expiry and code_expiry:
        printed_expiry = code_expiry
    if raw_code and (not req.batch or req.batch == raw_code) and code_batch:
        printed_batch = code_batch

    batch_status = "match" if (printed_batch and code_batch and printed_batch == code_batch) else (
        "mismatch" if (printed_batch and code_batch) else "not_comparable"
    )
    expiry_status = "match" if (printed_expiry and code_expiry and printed_expiry == code_expiry) else (
        "mismatch" if (printed_expiry and code_expiry) else "not_comparable"
    )

    result = ScanResult(
        id=scan_id,
        ts=ts,
        label=fusion.label,
        partial=True,
        quality=None,
        ocr=OcrResult(
            ran=False,
            name=OcrField(value=printed_name, confidence=0.95) if printed_name else None,
            batch=OcrField(value=printed_batch, confidence=0.95) if printed_batch else None,
            expiry=OcrField(value=printed_expiry, confidence=0.95) if printed_expiry else None,
            manufacturer=OcrField(value=req.manufacturer, confidence=0.90) if req.manufacturer else None,
        ),
        code=code_schema,
        visual=VisualResult(ran=False),
        authenticity=AuthenticityResult(
            ran=True,
            registry=RegistryStatus(status=registry_result.status, simulated=registry_result.simulated),
            alert=AlertMatch(
                match_level=alert_result.match_level,
                alert_id=alert_result.alert_id,
                source_month=alert_result.source_month,
                source_url=alert_result.source_url,
                snapshot_date=alert_result.snapshot_date,
                snapshot_stale=alert_result.snapshot_stale,
            ) if alert_result else None,
            clone=CloneSchema(
                signal=clone_result.signal,
                verdict_text=clone_result.verdict_text,
            ) if clone_result else None,
        ),
        reasons=[Reason(id=r["id"], text=r["text"], severity=r["severity"], contribution=r.get("contribution", 0)) for r in reasons_local],
        score=fusion.score,
        score_breakdown=ScoreBreakdown(**fusion.score_breakdown),
        checked=fusion.checked,
        not_checked=fusion.not_checked + ["Image capture — manual entry only", "Visual comparison — no image"],
        confidence="low",
        confidence_note="Manual entry — image and visual layers did not run",
        field_rows=[
            FieldRow(
                field="name",
                printed=printed_name,
                code_value=None,
                expected=fields.matched_medicine.get("name") if fields and fields.matched_medicine else None,
                status="match" if fields and fields.matched_medicine and printed_name and printed_name.strip().lower() == fields.matched_medicine.get("name", "").lower() else ("unclear" if fields and fields.matched_medicine else "not_comparable"),
            ),
            FieldRow(
                field="batch",
                printed=printed_batch,
                code_value=code_batch,
                expected=None,
                status=batch_status,
            ),
            FieldRow(
                field="expiry",
                printed=printed_expiry,
                code_value=code_expiry,
                expected=None,
                status=expiry_status,
            ),
        ] + ([
            FieldRow(
                field="manufacturer",
                printed=req.manufacturer or (fields.matched_medicine.get("manufacturer") if fields and fields.matched_medicine else None),
                code_value=None,
                expected=fields.matched_medicine.get("manufacturer") if fields and fields.matched_medicine else None,
                status="match" if fields and fields.matched_medicine and (req.manufacturer or fields.matched_medicine.get("manufacturer")) and (req.manufacturer or fields.matched_medicine.get("manufacturer", "")).strip().lower() == (fields.matched_medicine.get("manufacturer") or "").lower() else "not_comparable",
            )
        ] if req.manufacturer or (fields and fields.matched_medicine and fields.matched_medicine.get("manufacturer")) else []),
        next_steps=next_steps_local,
        simulated_flags=simulated_flags,
        disclaimer="Screening only. This does not confirm authenticity.",
        session_id=req.session_id,
    )

    _store_scan(db, result, req.device_token, session_id=req.session_id)
    return result


# ── GET /demo/{case_id} ───────────────────────────────────────────────────

DEMO_CASES = {
    "genuine", "expiry_tampered", "logo_shifted",
    "clone_code", "alert_match", "blurry"
}


@router.get("/demo/{case_id}", response_model=ScanResult)
async def get_demo(
    case_id: str,
    db: Session = Depends(get_db),
):
    """Run the real pipeline on a prebuilt demo case image."""
    if case_id not in DEMO_CASES:
        raise HTTPException(status_code=404, detail={"error": {"code": "NOT_FOUND", "message": "Unknown demo case"}})

    case_dir = DEMO_CASES_DIR / case_id
    pack_path = case_dir / "pack.jpg"
    code_path = case_dir / "code.png"

    img_bytes: Optional[bytes] = None
    img2_bytes: Optional[bytes] = None

    if pack_path.exists():
        img_bytes = pack_path.read_bytes()
    else:
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            candidates = list(case_dir.glob(ext))
            if candidates:
                img_bytes = candidates[0].read_bytes()
                break

    if code_path.exists():
        img2_bytes = code_path.read_bytes()

    if img_bytes is None:
        # Generate a simple placeholder image
        import numpy as np
        import cv2 as cv2_local
        placeholder = np.ones((480, 640, 3), dtype=np.uint8) * 200
        cv2_local.putText(placeholder, f"DEMO: {case_id}", (50, 240),
                          cv2_local.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 50), 2)
        _, buf = cv2_local.imencode(".jpg", placeholder)
        img_bytes = bytes(buf)

    result = await _run_full_pipeline(
        img_bytes=img_bytes,
        img2_bytes=img2_bytes,
        device_token="demo_device",
        lat=None,
        lng=None,
        language="en",
        db=db,
        extra_simulated_flags=["demo_case"],
    )
    return result
