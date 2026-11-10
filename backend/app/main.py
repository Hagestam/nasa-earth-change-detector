"""FastAPI application for the Africa Environmental Change Monitor."""

from __future__ import annotations

import logging
import os
from datetime import date
from typing import Any, Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from .services import analysis, interpretation, nasa_power


logger = logging.getLogger(__name__)
app = FastAPI(
    title="Africa Environmental Change Monitor API",
    version="0.1.0",
    description="NASA POWER precipitation ingestion, statistical trend analysis, and optional scientific interpretation.",
)
cors_origins = [
    origin.strip()
    for origin in os.environ.get(
        "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class BoundingBox(BaseModel):
    latitude_min: float = Field(ge=-90, le=90)
    latitude_max: float = Field(ge=-90, le=90)
    longitude_min: float = Field(ge=-180, le=180)
    longitude_max: float = Field(ge=-180, le=180)


class PrecipitationAnalysisRequest(BaseModel):
    start_date: date
    end_date: date
    bbox: BoundingBox
    period: Literal["monthly", "annual"] = "annual"
    coverage_threshold: float = Field(default=1.0, gt=0, le=1)
    alpha: float = Field(default=0.05, gt=0, lt=1)
    fdr_method: Literal["by", "bh"] = "by"
    n_resamples: int = Field(default=499, ge=19, le=9999)
    block_length: int | None = Field(default=None, ge=1)
    random_seed: int = 0
    min_observations: int = Field(default=10, ge=2)
    min_observations_per_season: int = Field(default=3, ge=1)
    min_seasons: int = Field(default=10, ge=1, le=12)
    include_interpretation: bool = False


def _json_safe(value: Any) -> Any:
    """Convert Pandas/NumPy values and missing values into strict JSON types."""
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, date)):
        return value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if pd.isna(value):
        return None
    return value


def _request_dict(request: PrecipitationAnalysisRequest) -> dict[str, Any]:
    # Support Pydantic v1 and v2 without using the v2-deprecated dict method.
    dump = getattr(request, "model_dump", None)
    values = dump() if callable(dump) else request.dict()
    return _json_safe(values)


def _run_pipeline(request: PrecipitationAnalysisRequest) -> dict[str, Any]:
    """Execute the synchronous services away from the ASGI event loop."""
    bbox = request.bbox
    daily = nasa_power.fetch_regional_precipitation(
        start_date=request.start_date,
        end_date=request.end_date,
        latitude_min=bbox.latitude_min,
        latitude_max=bbox.latitude_max,
        longitude_min=bbox.longitude_min,
        longitude_max=bbox.longitude_max,
    )

    validation_rows: list[dict[str, Any]] = []
    for (latitude, longitude), cell in daily.groupby(["latitude", "longitude"], sort=True):
        diagnostics = analysis.validate_environmental_data(
            cell[["time", "precipitation"]],
            expected_frequency="daily",
            value_column="precipitation",
            timestamp_column="time",
        )
        validation_rows.append({"latitude": float(latitude), "longitude": float(longitude), **diagnostics})

    if request.period == "annual":
        # Compose the existing daily->monthly and monthly->annual paths. The
        # engine's daily->annual branch relies on a Pandas Period property that
        # is absent in supported current Pandas releases.
        monthly = nasa_power.aggregate_regional_precipitation(
            daily,
            period="monthly",
            coverage_threshold=request.coverage_threshold,
        )
        aggregated = nasa_power.aggregate_regional_precipitation(
            monthly,
            period="annual",
            source_frequency="monthly",
            coverage_threshold=1.0,
        )
    else:
        aggregated = nasa_power.aggregate_regional_precipitation(
            daily,
            period="monthly",
            coverage_threshold=request.coverage_threshold,
        )
    results = analysis.analyze_spatial_trends(
        aggregated[["time", "latitude", "longitude", "precipitation"]],
        temporal_frequency=request.period,
        alpha=request.alpha,
        fdr_method=request.fdr_method,
        n_resamples=request.n_resamples,
        block_length=request.block_length,
        random_seed=request.random_seed,
        min_observations=request.min_observations,
        min_observations_per_season=request.min_observations_per_season,
        min_seasons=request.min_seasons,
        units=aggregated.attrs.get("units"),
        source_provenance=aggregated.attrs.get("source_provenance", {}),
    )
    summary = analysis.generate_analysis_summary(results)
    results_records = _json_safe(results.to_dict(orient="records"))
    validation_summary = {
        "cells_checked": len(validation_rows),
        "cells_with_errors": sum(not row["valid"] for row in validation_rows),
        "missing_value_count": sum(int(row.get("missing_value_count", 0) or 0) for row in validation_rows),
        "invalid_value_count": sum(int(row.get("invalid_value_count", 0) or 0) for row in validation_rows),
        "cells": validation_rows,
    }
    response: dict[str, Any] = {
        "request": _request_dict(request),
        "data": {
            "units": aggregated.attrs.get("units"),
            "source_provenance": aggregated.attrs.get("source_provenance", {}),
            "aggregation": aggregated.attrs.get("aggregation", {}),
            "daily_observation_count": int(len(daily)),
            "aggregated_observation_count": int(len(aggregated)),
        },
        "validation": validation_summary,
        "analysis": {
            "period": request.period,
            "units": aggregated.attrs.get("units"),
            "summary": summary,
            "testing_family": results.attrs.get("testing_family", {}),
            "analysis_settings": results.attrs.get("analysis_settings", {}),
            "results": results_records,
        },
        "interpretation": None,
    }
    if request.include_interpretation:
        try:
            response["interpretation"] = {
                "status": "completed",
                "result": interpretation.interpret_analysis_results(
                    results,
                    context={
                        "summary": summary,
                        "period": request.period,
                        "units": aggregated.attrs.get("units"),
                        "source_provenance": aggregated.attrs.get("source_provenance", {}),
                        "date_range": {
                            "start": request.start_date.isoformat(),
                            "end": request.end_date.isoformat(),
                        },
                    },
                ),
            }
        except interpretation.InterpretationError as exc:
            logger.info("Optional interpretation stage failed: %s", type(exc).__name__)
            response["interpretation"] = {
                "status": "unavailable",
                "error": str(exc),
            }
    return _json_safe(response)


@app.get("/health")
async def health() -> dict[str, str]:
    """Report that the API process is ready to accept requests."""
    return {"status": "ok"}


@app.post("/api/v1/analysis/precipitation")
async def analyze_precipitation(request: PrecipitationAnalysisRequest) -> dict[str, Any]:
    """Fetch, aggregate, and analyze regional precipitation trends."""
    if request.start_date > request.end_date:
        raise HTTPException(status_code=422, detail="start_date must be on or before end_date.")
    try:
        return await run_in_threadpool(_run_pipeline, request)
    except nasa_power.NasaPowerRequestError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except nasa_power.NasaPowerResponseError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Precipitation analysis pipeline failed")
        raise HTTPException(status_code=500, detail="The precipitation analysis pipeline failed.") from exc
