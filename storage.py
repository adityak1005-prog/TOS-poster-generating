import os
import boto3

BUCKET = os.environ.get("S3_BUCKET", "movie-booth-uploads")
s3 = boto3.client("s3")


def upload(data: bytes, key: str) -> str:
    s3.put_object(Bucket=BUCKET, Key=key, Body=data, ContentType="image/jpeg", ACL="public-read")
    region = s3.meta.region_name
    return f"https://{BUCKET}.s3.{region}.amazonaws.com/{key}"
