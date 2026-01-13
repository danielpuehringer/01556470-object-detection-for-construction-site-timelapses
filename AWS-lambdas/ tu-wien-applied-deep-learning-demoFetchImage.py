# this file is deployed via AWS Lambda to fetch an image from a given URL

import boto3
import os
import urllib.parse

s3 = boto3.client("s3")
BUCKET = os.environ["BUCKET_NAME"]
REGION = os.environ.get("AWS_REGION", "eu-central-1")

def s3_public_url(bucket: str, key: str, region: str) -> str:
    # Use virtual-hosted–style URL. us-east-1 is a special case.
    key_enc = urllib.parse.quote(key, safe="/~!*'();:@&=+$,?%#[]")  # keep slashes
    if region == "us-east-1":
        return f"https://{bucket}.s3.amazonaws.com/{key_enc}"
    return f"https://{bucket}.s3.{region}.amazonaws.com/{key_enc}"

def lambda_handler(event, context):
    qs = event.get("queryStringParameters") or {}
    key = qs.get("key")
    if not key:
        return {"statusCode": 400, "body": "Missing query parameter: key"}

    # Optional: check it exists (nice for 404 behavior)
    try:
        s3.head_object(Bucket=BUCKET, Key=key)
    except s3.exceptions.NoSuchKey:
        return {"statusCode": 404, "body": "Not found"}
    except Exception as e:
        # If you don't want to reveal details, replace with a generic message.
        return {"statusCode": 500, "body": f"Error: {str(e)}"}

    url = s3_public_url(BUCKET, key, REGION)

    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*"
        },
        "body": f'{{"url":"{url}"}}'
    }