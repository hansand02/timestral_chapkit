"""ML service for chapkit_timestral_tsfm."""

# =====================================================================
# What is this file?
# =====================================================================
# main.py defines your chapkit service. The bits you typically edit are:
#   1. The Config class below (the parameters your model takes).
#   2. The on_train and on_predict functions (your actual ML logic).
#   3. The ServiceInfo block (your name, email, model description).
#
# Everything else (database wiring, HTTP routing, async plumbing) is
# boilerplate that chapkit needs and you almost never have to touch.
# =====================================================================

import asyncio
import os
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog
from chapkit import BaseConfig
from chapkit.api import (
    AssessedStatus,
    MLServiceBuilder,
    MLServiceInfo,
    ModelMetadata,
    PeriodType,
)
from chapkit.artifact import ArtifactHierarchy
from chapkit.data import DataFrame
from chapkit.ml import FunctionalModelRunner
from geojson_pydantic import FeatureCollection
from timestral import TimestralForecaster

log = structlog.get_logger()

MODEL_ID = "mistralai/Timestral-209M"
OUTPUT_LEVELS = np.linspace(0.01, 0.99, 100)  # written as sample_0..sample_99
MAX_CONTEXT = 32_768  # Timestral's trained context length (time steps)
MAX_HORIZON = 1_024  # Timestral's max forecast length (time steps)


class ChapkitTimestralTsfmConfig(BaseConfig):
    """Configuration for chapkit_timestral_tsfm."""

    # Fallback horizon; chapkit overrides it per request.
    prediction_periods: int = 3


@cache
def load_model(model_id: str) -> TimestralForecaster:
    # Loads from the Hugging Face cache (downloads on first use).
    return TimestralForecaster.from_pretrained(model_id)


async def on_train(
    config: ChapkitTimestralTsfmConfig,
    data: DataFrame,
    geo: FeatureCollection | None = None,
) -> Any:
    """Zero-shot: nothing to train. Download the weights and return only the model id,
    since chapkit pickles whatever is returned into its database."""
    await asyncio.to_thread(load_model, MODEL_ID)  # off the event loop, so polls still get answered
    log.info("model_ready", model_id=MODEL_ID)
    return MODEL_ID


async def on_predict(
    config: ChapkitTimestralTsfmConfig,
    model: Any,
    historic: DataFrame,
    future: DataFrame,
    geo: FeatureCollection | None = None,
) -> DataFrame:
    """Forecast with Timestral. Covariates come from the config's additional_continuous_covariates
    (known over past and future); none means target history only."""
    forecaster = await asyncio.to_thread(load_model, model)
    past = historic.to_pandas()
    future_df = future.to_pandas()
    both = pd.concat([past, future_df])

    # Forecast log(1 + cases) so outbreak peaks are not huge outliers; undone with expm1 below.
    # The upper quantiles blow up when transformed back, so cap them at 10x the location's record.
    past["disease_cases"] = np.log1p(past["disease_cases"])
    context = past.pivot(
        index="location", columns="time_period", values="disease_cases"
    )
    context = context.iloc[:, -MAX_CONTEXT:]  # keep the most recent steps
    cap = np.log1p(10 * np.expm1(context.max(axis=1).fillna(0).values))
    locations = context.index
    future_periods = sorted(future_df["time_period"].unique())
    assert len(future_periods) <= MAX_HORIZON, (
        f"horizon {len(future_periods)} > {MAX_HORIZON}"
    )
    periods = list(context.columns) + future_periods
    covariates = config.additional_continuous_covariates
    known = (
        np.stack(
            [
                both.pivot(index="location", columns="time_period", values=c)
                .loc[locations, periods]
                .values
                for c in covariates
            ],
            axis=1,
        )  # (location, covariate, past + future periods)
        if covariates
        else None
    )

    # Slow, CPU-bound torch call: run it in a thread so the server keeps answering polls and /health.
    forecast = await asyncio.to_thread(
        forecaster.forecast, context.values, prediction_length=len(future_periods), known_covariates=known
    )
    quantiles = np.asarray(forecast.quantiles)[
        :, 0
    ]  # (location, horizon, quantile levels)
    quantiles = np.expm1(np.clip(quantiles, 0, cap[:, None, None]))

    rows = []
    for i, location in enumerate(locations):
        for h, period in enumerate(future_periods):
            samples = np.interp(
                OUTPUT_LEVELS, forecast.quantile_levels, quantiles[i, h]
            )
            rows.append([location, period, *samples])
    columns = ["location", "time_period"] + [
        f"sample_{i}" for i in range(len(OUTPUT_LEVELS))
    ]
    log.info("predictions_made", rows=len(rows))
    return DataFrame.from_pandas(pd.DataFrame(rows, columns=columns))


# Create ML service info with metadata
info = MLServiceInfo(
    id="chapkit-timestral-tsfm",
    display_name="chapkit_timestral_tsfm",
    version="0.1.0",
    description="An implementation of the timestral foundation model for time series https://huggingface.co/mistralai/Timestral-209M",
    model_metadata=ModelMetadata(
        author="Hans Andersen",
        # Maturity colour shown in chap-core's UI. Pick conservatively:
        # green=validated for production, yellow=ready for rigorous testing,
        # orange=promising on limited data, red=highly experimental,
        # gray=deprecated. See README.md for full meanings
        author_note="Promising results on THA+LAO+VNM test suite",
        author_assessed_status=AssessedStatus.orange,
        contact_email="hans.andersen02@gmail.com",
    ),
    period_type=PeriodType.monthly,
    allow_free_additional_continuous_covariates=True,
    min_prediction_periods=0,
    max_prediction_periods=MAX_HORIZON,
)

# Create artifact hierarchy for ML artifacts
HIERARCHY = ArtifactHierarchy(
    name="chapkit_timestral_tsfm",
    level_labels={0: "ml_training_workspace", 1: "ml_prediction"},
)

# Create functional model runner
runner: FunctionalModelRunner[ChapkitTimestralTsfmConfig] = FunctionalModelRunner(
    on_train=on_train,
    on_predict=on_predict,
)

# Database configuration
# Uses environment variable or defaults to data/chapkit.db
# Creates data directory if it doesn't exist
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///data/chapkit.db")
if DATABASE_URL.startswith("sqlite") and ":///" in DATABASE_URL:
    db_path = Path(DATABASE_URL.split("///")[1])
    db_path.parent.mkdir(parents=True, exist_ok=True)

# Build the FastAPI application
app = (
    MLServiceBuilder(
        info=info,
        config_schema=ChapkitTimestralTsfmConfig,
        hierarchy=HIERARCHY,
        runner=runner,
        database_url=DATABASE_URL,
    )
    .with_monitoring()
    # See compose.yml for chap-core self-registration env vars.
    .with_registration()
    .build()
)


if __name__ == "__main__":
    from chapkit.api import run_app

    # Default to port 9090 to match the Docker compose host port and avoid the usual
    # busy ports (8000, 8080). Override with the PORT env var. Set reload=True to
    # enable hot reloading during development.
    run_app("main:app", reload=False, port=9090)
