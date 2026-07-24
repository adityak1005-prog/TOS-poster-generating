import os
from supabase import create_client

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
SUPABASE_BUCKET = os.environ.get("SUPABASE_BUCKET")

_client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --------------------------------------------------------------------------
# Fallback storage target: instead of an auto-delete sweep to keep the
# primary bucket under its free-tier limit, upload() now fails over to a
# second bucket (optionally a second Supabase project entirely) if the
# primary upload raises for any reason -- quota exceeded, transient network
# error, bad primary credentials, whatever. All three fallback env vars are
# optional and independent:
#   - SUPABASE_BUCKET_FALLBACK alone: same project, different bucket name
#     (note: on Supabase's free tier, storage quota is per-PROJECT, not per
#     bucket, so this alone doesn't add real capacity -- it only helps if
#     the primary bucket itself is misconfigured/deleted/renamed).
#   - SUPABASE_URL_FALLBACK + SUPABASE_KEY_FALLBACK + (optionally)
#     SUPABASE_BUCKET_FALLBACK: a genuinely separate Supabase project, which
#     does add independent capacity.
# If none of these are set, upload() behaves exactly as before -- a failure
# just raises, same as always.
# --------------------------------------------------------------------------
FALLBACK_URL = os.environ.get("SUPABASE_URL_FALLBACK") or SUPABASE_URL
FALLBACK_KEY = os.environ.get("SUPABASE_KEY_FALLBACK") or SUPABASE_KEY
FALLBACK_BUCKET = os.environ.get("SUPABASE_BUCKET_FALLBACK")

_fallback_client = None
if FALLBACK_BUCKET:
    _fallback_client = (
        _client if (FALLBACK_URL, FALLBACK_KEY) == (SUPABASE_URL, SUPABASE_KEY)
        else create_client(FALLBACK_URL, FALLBACK_KEY)
    )


def _upload_via(client, bucket: str, file_bytes: bytes, file_name: str) -> str:
    client.storage.from_(bucket).upload(
        path=file_name,
        file=file_bytes,
        file_options={"upsert": "true"}
    )
    return client.storage.from_(bucket).get_public_url(file_name)


def upload(file_bytes: bytes, file_name: str) -> str:
    """Uploads a file to the primary Supabase bucket and returns the public
    URL. Falls back to a secondary bucket/project (see above) if the
    primary upload raises and a fallback is configured -- if not configured,
    or if the fallback also fails, the original exception propagates so the
    caller's own error handling (and terminal logging) still applies."""
    try:
        return _upload_via(_client, SUPABASE_BUCKET, file_bytes, file_name)
    except Exception as e:
        if not _fallback_client:
            raise
        print(f"[storage] primary bucket upload failed ({e}); "
              f"retrying against fallback bucket '{FALLBACK_BUCKET}'")
        try:
            return _upload_via(_fallback_client, FALLBACK_BUCKET, file_bytes, file_name)
        except Exception as e2:
            print(f"[storage] fallback bucket upload also failed ({e2})")
            raise


def list_objects() -> list:
    """Returns the raw object listing for the primary bucket (each entry
    includes name/created_at/etc.) -- used by the on-demand
    /booth/admin/cleanup sweep. Doesn't touch upload() or its behavior."""
    return _client.storage.from_(SUPABASE_BUCKET).list()


def delete_objects(file_names: list) -> None:
    """Deletes the given file names from the primary bucket. No-op on an
    empty list. Used by the on-demand /booth/admin/cleanup sweep."""
    if not file_names:
        return
    _client.storage.from_(SUPABASE_BUCKET).remove(file_names)