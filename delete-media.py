import boto3
from dotenv import load_dotenv
import os

load_dotenv()

s3 = boto3.client(
    "s3",
    endpoint_url=f"https://{os.environ['R2_PROD_ACCOUNT_ID']}.r2.cloudflarestorage.com",
    aws_access_key_id=os.environ["R2_PROD_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["R2_PROD_SECRET_ACCESS_KEY"],
)

bucket = "nadeshiko-production"
prefix = "media/7674/"


# List and delete in batches of 1000
paginator = s3.get_paginator("list_objects_v2")
for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
    objects = page.get("Contents", [])
    if objects:
        delete_keys = [{"Key": obj["Key"]} for obj in objects]
        s3.delete_objects(Bucket=bucket, Delete={"Objects": delete_keys})
        print(f"Deleted {len(delete_keys)} objects")

print("Done")


