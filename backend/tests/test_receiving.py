"""
tests/test_receiving.py — Receiving mode end-to-end and conflict tests.
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Create tables before importing app
from app.db import engine, Base
Base.metadata.create_all(bind=engine)

from app.main import app

client = TestClient(app)


class TestReceivingMode:
    def test_session_lifecycle_and_batch_conflict(self):
        """
        Create a receiving session, scan two manual entries with the same batch
        but different expiry dates, and assert that the summary reports an expiry conflict.
        """
        # 1. Create a receiving session
        create_res = client.post("/receiving/sessions", json={
            "supplier": "Apex Pharma Logistics",
            "invoice_no": "INV-2026-TEST",
        })
        assert create_res.status_code == 201
        session_data = create_res.json()
        assert "id" in session_data
        session_id = session_data["id"]

        # 2. Add first pack: Batch B-999, Expiry 11/2026
        pack1_res = client.post(f"/receiving/sessions/{session_id}/items", json={
            "name": "Augmentin 625 Duo",
            "batch": "B-999",
            "expiry": "11/2026",
            "manufacturer": "GSK",
            "device_token": "rec_tester_01",
        })
        assert pack1_res.status_code == 201
        scan1_id = pack1_res.json()["id"]

        # 3. Add second pack: same Batch B-999, but conflicting Expiry 05/2025
        pack2_res = client.post(f"/receiving/sessions/{session_id}/items", json={
            "name": "Augmentin 625 Duo",
            "batch": "B-999",
            "expiry": "05/2025",
            "manufacturer": "GSK",
            "device_token": "rec_tester_01",
        })
        assert pack2_res.status_code == 201

        # 4. Fetch summary and assert conflict detection
        summary_res = client.get(f"/receiving/sessions/{session_id}/summary")
        assert summary_res.status_code == 200
        summary = summary_res.json()

        assert summary["scanned"] == 2
        assert summary["flagged"] >= 1
        assert len(summary["batch_conflicts"]) >= 1

        conflict = summary["batch_conflicts"][0]
        assert conflict["batch"] == "B-999"
        assert conflict["conflict_type"] == "expiry_mismatch"
        assert "11/2026" in conflict["detail"]
        assert "05/2025" in conflict["detail"]

        # 5. Test PDF export for session
        pdf_res = client.get(f"/receiving/sessions/{session_id}/export?format=pdf")
        assert pdf_res.status_code == 200
        assert pdf_res.headers["content-type"] == "application/pdf"
        assert pdf_res.content.startswith(b"%PDF")

        # 6. Test CSV export for session
        csv_res = client.get(f"/receiving/sessions/{session_id}/export?format=csv")
        assert csv_res.status_code == 200
        assert "text/csv" in csv_res.headers["content-type"]
        assert b"B-999" in csv_res.content

        # 7. Test single scan PDF endpoint
        scan_pdf_res = client.get(f"/scans/{scan1_id}/report.pdf")
        assert scan_pdf_res.status_code == 200
        assert scan_pdf_res.headers["content-type"] == "application/pdf"
        assert scan_pdf_res.content.startswith(b"%PDF")
