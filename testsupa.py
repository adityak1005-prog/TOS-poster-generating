import os
from dotenv import load_dotenv
from supabase import create_client

# Load variables from .env
load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
SUPABASE_BUCKET = os.environ.get("SUPABASE_BUCKET")

print("--- SUPABASE CONNECTION TEST ---")
print(f"URL:    {SUPABASE_URL}")
print(f"Bucket: {SUPABASE_BUCKET}\n")

if not SUPABASE_URL or not SUPABASE_KEY or not SUPABASE_BUCKET:
    print("❌ Error: Missing env variables in .env file.")
    exit(1)

try:
    # 1. Initialize client
    client = create_client(SUPABASE_URL, SUPABASE_KEY)

    # 2. Upload dummy text bytes
    test_filename = "test_ping.txt"
    test_bytes = b"Supabase storage upload test successful!"

    print(f"Uploading '{test_filename}' to bucket '{SUPABASE_BUCKET}'...")
    client.storage.from_(SUPABASE_BUCKET).upload(
        path=test_filename,
        file=test_bytes,
        file_options={"upsert": "true"}
    )
    print("✅ Upload succeeded!")

    # 3. Retrieve public URL
    public_url = client.storage.from_(SUPABASE_BUCKET).get_public_url(test_filename)
    print(f"✅ Public URL generated: {public_url}")
    print("\n🎉 Your Supabase credentials & bucket settings are 100% correct!")

except Exception as err:
    print("\n❌ TEST FAILED:")
    print(err)