"""
============================================================
app/models/biometrics.py — Layer 4 Biometric Stream Model
============================================================
PURPOSE:
    Maps to the `biometrics_stream` table.
    Phase 1 creates this model so all FK relationships work.
    Phase 10 builds the full ingestion API and SQL aggregate views.

CONNECTED TO:
    Phase 1  → Table exists for schema completeness
    Phase 10 → POST /api/v1/biometrics fills this table
    Phase 10 → SQL views (hourly, daily) aggregate this data
    Phase 8  → Cascade deleted on Right-to-Forget
============================================================
"""
from sqlalchemy import Column, String, DateTime, Float, SmallInteger, func
from app.core.database import Base


class BiometricsStream(Base):
    """
    Layer 4 Biometric Stream — time-series wearable data row.

    metric_type values: resting_hr, hrv_rmssd, sleep_deep_min,
                        sleep_rem_min, steps, active_calories, spo2
    quality_flag      : 1=good, 0=noisy/artifact (filtered in Phase 10 views)
    """
    __tablename__ = "biometrics_stream"

    id = Column(String, primary_key=True, server_default="gen_random_uuid()")
    time = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), index=True)
    user_id = Column(String, nullable=False, index=True)
    metric_type = Column(String(32), nullable=False)
    value = Column(Float, nullable=False)
    device_id = Column(String(64), nullable=True)
    quality_flag = Column(SmallInteger, default=1)
