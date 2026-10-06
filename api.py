```python
import os
import json
import uuid
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field


app = FastAPI(
    title="SentrySearch API",
    description="Video indexing and semantic search API",
    version="1.0.0",
)

# Railway persistent volume should be mounted here.
DATA_DIR = Path(os.getenv("SENTRYSEARCH_DATA_DIR", "/data"))
TEMP_DIR = Path("/tmp/sentrysearch")

TEMP_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

API_KEY = os.getenv("API_KEY")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class IndexRequest(BaseModel):
    url: str = Field(..., description="Public S3 URL or presigned S3 URL")
    video_id: Optional[str] = None
    metadata: dict = Field(default_factory=dict)


class SearchRequest(BaseModel):
    query: str
    limit: int = Field(default=10, ge=1, le=100)


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

def authenticate(x_api_key: Optional[str]):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def download_video(url: str, destination: Path):
    """
    Download a video from an HTTP/S3 URL to a temporary file.
    """
    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=httpx.Timeout(
                connect=30,
                read=300,
                write=300,
                pool=30,
            ),
        ) as client:

            async with client.stream("GET", url) as response:
                response.raise_for_status()

                with destination.open("wb") as f:
                    async for chunk in response.aiter_bytes(chunk_size=1024 * 1024):
                        f.write(chunk)

    except httpx.HTTPError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not download video: {str(e)}",
        )


def run_sentrysearch(args: list[str]):
    """
    Execute the installed SentrySearch CLI.

    Example:
        sentrysearch index /tmp/video.mp4
    """
    command = ["sentrysearch"] + args

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=1800,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"SentrySearch failed:\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    return result.stdout


def save_metadata(video_id: str, data: dict):
    """
    Keep our own metadata alongside the SentrySearch index.

    This lets us map search results back to the original S3 object.
    """
    metadata_dir = DATA_DIR / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    metadata_file = metadata_dir / f"{video_id}.json"

    metadata_file.write_text(
        json.dumps(data, indent=2),
        encoding="utf-8",
    )


def load_all_metadata():
    metadata_dir = DATA_DIR / "metadata"

    if not metadata_dir.exists():
        return []

    results = []

    for path in metadata_dir.glob("*.json"):
        try:
            results.append(
                json.loads(path.read_text(encoding="utf-8"))
            )
        except Exception:
            pass

    return results


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "data_dir": str(DATA_DIR),
    }


@app.post("/index")
async def index_video(
    request: IndexRequest,
    x_api_key: Optional[str] = Header(default=None),
):
    """
    Download a video and add it to SentrySearch.

    Example:

    POST /index

    {
        "url": "https://bucket.s3.amazonaws.com/reels/video.mp4",
        "video_id": "video_123",
        "metadata": {
            "brand": "Mamaearth",
            "platform": "instagram"
        }
    }
    """

    authenticate(x_api_key)

    video_id = request.video_id or str(uuid.uuid4())

    # Keep the extension from the URL if possible.
    extension = ".mp4"

    try:
        suffix = Path(request.url.split("?")[0]).suffix
        if suffix:
            extension = suffix
    except Exception:
        pass

    video_path = TEMP_DIR / f"{video_id}{extension}"

    try:
        # ---------------------------------------------------------------
        # Download
        # ---------------------------------------------------------------

        await download_video(
            request.url,
            video_path,
        )

        if not video_path.exists() or video_path.stat().st_size == 0:
            raise HTTPException(
                status_code=400,
                detail="Downloaded video is empty.",
            )

        # ---------------------------------------------------------------
        # Index using SentrySearch
        # ---------------------------------------------------------------

        output = run_sentrysearch(
            [
                "index",
                str(video_path),
            ]
        )

        # ---------------------------------------------------------------
        # Save our own metadata
        # ---------------------------------------------------------------

        save_metadata(
            video_id,
            {
                "video_id": video_id,
                "url": request.url,
                "metadata": request.metadata,
                "filename": video_path.name,
                "indexed_at": __import__("datetime")
                .datetime.utcnow()
                .isoformat()
                + "Z",
            },
        )

        return {
            "success": True,
            "video_id": video_id,
            "url": request.url,
            "sentrysearch_output": output,
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )

    finally:
        # ---------------------------------------------------------------
        # Delete temporary video
        # ---------------------------------------------------------------

        if video_path.exists():
            try:
                video_path.unlink()
            except Exception:
                pass


@app.post("/search")
def search_videos(
    request: SearchRequest,
    x_api_key: Optional[str] = Header(default=None),
):
    """
    Semantic search over indexed videos.

    Example:

    POST /search

    {
        "query": "woman applying sunscreen",
        "limit": 10
    }
    """

    authenticate(x_api_key)

    try:
        output = run_sentrysearch(
            [
                "search",
                request.query,
            ]
        )

        return {
            "query": request.query,
            "results": output,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


@app.get("/videos")
def list_videos(
    x_api_key: Optional[str] = Header(default=None),
):
    """
    List videos that have been indexed through this API.
    """

    authenticate(x_api_key)

    return {
        "videos": load_all_metadata(),
    }


@app.delete("/videos/{video_id}")
def delete_video(
    video_id: str,
    x_api_key: Optional[str] = Header(default=None),
):
    """
    Remove our metadata for a video.

    Note:
    This does not remove the corresponding vector from ChromaDB because
    SentrySearch's CLI does not currently expose a stable delete interface.
    """

    authenticate(x_api_key)

    metadata_file = DATA_DIR / "metadata" / f"{video_id}.json"

    if not metadata_file.exists():
        raise HTTPException(
            status_code=404,
            detail="Video not found.",
        )

    metadata_file.unlink()

    return {
        "success": True,
        "video_id": video_id,
    }
```
