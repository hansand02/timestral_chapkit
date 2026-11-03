# chapkit_timestral_tsfm

ML service for chapkit_timestral_tsfm

This project was scaffolded using the [Chapkit](https://dhis2-chap.github.io/chapkit) CLI.

## What you'll edit

You typically only edit two places:

1. The `on_train` and `on_predict` functions in `main.py` - your model logic
2. The `Config` class and `MLServiceInfo` block in `main.py` - the parameters your
   model accepts and the metadata that describes it

Everything else (`Dockerfile`, `compose.yml`, the database plumbing) is wiring you
can leave alone.

## What you need installed

- **Python 3.13 + uv**. Install uv from [astral.sh/uv](https://docs.astral.sh/uv/).
- **Docker** with the compose plugin (for the production-ish path).

## Quick Start

### Development Mode

Install dependencies and run the service locally:

```bash
uv sync
uv run python main.py
```

The API will be available at http://localhost:9090

### Docker

Build and run with Docker Compose:

```bash
uv sync                  # one-time, generates uv.lock that the Dockerfile pins against
docker compose up --build
```

The API will be available at:
- API: http://localhost:9090
- API Docs: http://localhost:9090/docs

The compose stack maps host port 9090 to container port 8000. Edit `compose.yml`
to change the host port. Local dev (`python main.py`) also defaults to 9090.

The container runs as an unprivileged `chapkit` user on a read-only root
filesystem, the same posture chap-core uses for its own services. The only
writable paths are `/work/data` (the SQLite database, a named volume) and `/tmp`
(ML workspaces, a tmpfs held in RAM). If your model writes anywhere else, add a
volume for that path in `compose.yml` or drop `read_only: true`; if it produces
large intermediate files, raise the tmpfs `size`.

### Verify the service

Once the service is running (locally or in Docker), exercise the full
config -> train -> predict flow with `chapkit test`:

```bash
# Against the running service (default http://localhost:9090)
uv run chapkit test

# More aggressive: 3 configs, 2 trainings each, 2 predictions per model
uv run chapkit test -c 3 -t 2 -p 2 --verbose

# Auto-start an in-memory service for the duration of the test
uv run chapkit test --start-service
```

Note: this is an end-to-end **smoke test** of the API surface, not a model
quality test. `chapkit test` synthesises random training and prediction
data, drives the service through the full config -> train -> predict
lifecycle, and confirms each step returns a valid response. It does not
validate that your model produces meaningful predictions - only that the
service plumbing (config CRUD, the train job, artifact persistence, and
predict) all work. Handy after editing `main.py` or rebuilding the image.

## API Endpoints

The interactive Swagger UI at <http://localhost:9090/docs> is the easiest way to
poke at the service - it lets you fill in payloads and hit endpoints from the
browser without any extra tools.

If you prefer Postman or Insomnia, point them at the OpenAPI spec at
<http://localhost:9090/openapi.json> (Postman: *Import → Link* and paste the URL;
it generates a fresh collection). Re-import after API changes.

### Health Check

```bash
curl http://localhost:9090/health
```

### Configuration Management

Create a configuration. The response contains the config `id` used by `$train`:

```bash
curl -X POST http://localhost:9090/api/v1/configs \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my-config",
    "data": {"prediction_periods": 3}
  }'
```

### ML Operations

Training and prediction data are chapkit `DataFrame` payloads: an object with a
`columns` list and a `data` list of rows, where each row has one value per
column. Both `$train` and `$predict` return `202 Accepted` with a `job_id` and
the `artifact_id` the result will be stored under - the work itself runs in the
background, so you poll the job.

Train a model (substitute the config id from the previous step):

```bash
curl -X POST http://localhost:9090/api/v1/ml/\$train \
  -H "Content-Type: application/json" \
  -d '{
    "config_id": "YOUR_CONFIG_ID",
    "data": {
      "columns": ["time_period", "location", "rainfall", "mean_temperature", "disease_cases"],
      "data": [
        ["2023-01", "loc-1", 10.0, 25.0, 5.0],
        ["2023-02", "loc-1", 15.0, 28.0, 8.0],
        ["2023-03", "loc-1", 8.0, 22.0, 3.0],
        ["2023-04", "loc-1", 20.0, 30.0, 12.0]
      ]
    }
  }'
```

Poll the training job until `status` is `completed` (a failing training script
ends the job as `failed`, with the exit code and the diagnostic artifact id in
`error`):

```bash
curl http://localhost:9090/api/v1/jobs/YOUR_TRAIN_JOB_ID
```

Make predictions, passing the `artifact_id` returned by `$train`. `historic` is
the context the model has already seen, `future` the periods to predict:

```bash
curl -X POST http://localhost:9090/api/v1/ml/\$predict \
  -H "Content-Type: application/json" \
  -d '{
    "artifact_id": "YOUR_TRAIN_ARTIFACT_ID",
    "historic": {
      "columns": ["time_period", "location", "rainfall", "mean_temperature", "disease_cases"],
      "data": [
        ["2023-03", "loc-1", 8.0, 22.0, 3.0],
        ["2023-04", "loc-1", 20.0, 30.0, 12.0]
      ]
    },
    "future": {
      "columns": ["time_period", "location", "rainfall", "mean_temperature"],
      "data": [
        ["2023-05", "loc-1", 12.0, 26.0],
        ["2023-06", "loc-1", 18.0, 29.0]
      ]
    }
  }'
```

Poll the prediction job, then fetch the predictions from the artifact id that
`$predict` returned:

```bash
curl http://localhost:9090/api/v1/jobs/YOUR_PREDICT_JOB_ID
curl http://localhost:9090/api/v1/artifacts/YOUR_PREDICT_ARTIFACT_ID
```

## Customization

### Service identity (`MLServiceInfo`)

Open `main.py` and fill in the `MLServiceInfo` block - chap-core surfaces these
fields in its UI when listing models:

- `id`, `display_name`, `version`, `description` - human-readable identity.
- `model_metadata.author`, `contact_email`, `organization` - who owns it.
- `model_metadata.author_assessed_status` - your honest read of how validated
  the model is. Pick conservatively; chap-core shows this colour next to your
  model in the catalogue.

| Status | Meaning |
| --- | --- |
| `green` | Validated and ready for production use |
| `yellow` | Ready for more rigorous testing on diverse data |
| `orange` | Shows promise on limited data, needs manual configuration and careful evaluation |
| `red` | Highly experimental prototype, not validated, only for early experimentation |
| `gray` | Not intended for use - deprecated or kept only for backwards compatibility |

The scaffold ships `AssessedStatus.yellow`; bump it up or down to match reality.

Three provenance fields are filled in automatically and show up on
`GET /api/v1/info` next to the identity above:

- `git_revision` - the commit the image was built from. Read from the
  `GIT_REVISION` environment variable, which the `Dockerfile` takes as a build
  argument. The scaffolded `.github/workflows/publish-docker.yml` passes
  `github.sha`; locally, `GIT_REVISION=$(git rev-parse HEAD) docker compose up --build`
  or `docker build --build-arg GIT_REVISION=$(git rev-parse HEAD) .` does the same.
  Unset means `null`.
- `chapkit_version` and `servicekit_version` - the library versions installed in
  the image.

Set any of them explicitly on `MLServiceInfo` to override the automatic value.

### Update Model Configuration

Edit the configuration class in `main.py`:

```python
class ChapkitTimestralTsfmConfig(BaseConfig):
    # Required: number of prediction periods
    prediction_periods: int = 3
    # Add your parameters here
    min_samples: int = 5
    learning_rate: float = 0.01
```

### Customize Training

Edit the `on_train` function in `main.py` to implement your model training logic.

### Customize Prediction

Edit the `on_predict` function in `main.py` to implement your prediction logic.

## Project Structure

```
chapkit_timestral_tsfm/
├── main.py              # Main application file
├── pyproject.toml       # Python dependencies
├── Dockerfile           # Docker build configuration
├── compose.yml          # Docker Compose configuration
├── .dockerignore        # Files excluded from the Docker build context
├── .github/workflows/
│   └── publish-docker.yml  # Builds and publishes the image to GHCR
└── data/                # Database directory
    └── chapkit.db       # SQLite database (persisted)
```

## Documentation

- [Chapkit Documentation](https://dhis2-chap.github.io/chapkit)
- [FastAPI Documentation](https://fastapi.tiangolo.com)
- [Servicekit Documentation](https://dhis2-chap.github.io/servicekit)

## License

Add your license information here.