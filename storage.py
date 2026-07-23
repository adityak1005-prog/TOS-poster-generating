import os
from supabase import create_client

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
SUPABASE_BUCKET = os.environ.get("SUPABASE_BUCKET")

_client = create_client(SUPABASE_URL, SUPABASE_KEY)

def upload(file_bytes: bytes, file_name: str) -> str:
    """Uploads a file to Supabase and returns the public URL."""

    # Upload the file (upsert=true allows overwriting if testing the same filename)
    _client.storage.from_(SUPABASE_BUCKET).upload(
        path=file_name,
        file=file_bytes,
        file_options={"upsert": "true"}
    )

    # Return the direct public URL for the compose step
    return _client.storage.from_(SUPABASE_BUCKET).get_public_url(file_name)


def list_objects() -> list:
    """Returns the raw object listing for the bucket (each entry includes
    name/created_at/etc.) -- used by the cleanup sweep in app.py to find
    stale uploads. Doesn't touch upload() or its behavior."""
    return _client.storage.from_(SUPABASE_BUCKET).list()


def delete_objects(file_names: list) -> None:
    """Deletes the given file names from the bucket. No-op on an empty
    list. Used by the cleanup sweep in app.py."""
    if not file_names:
        return
    _client.storage.from_(SUPABASE_BUCKET).remove(file_names)