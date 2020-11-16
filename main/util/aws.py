import os
import boto3


def get_session():
    return boto3.Session(
        # hard-wired - change to env var when needed
        aws_access_key_id=os.getenv(''),
        aws_secret_access_key=os.getenv(''),
        region_name=os.getenv('AWS_DEFAULT_REGION')
    )


def build_s3():
    session = get_session()
    s3 = session.resource('s3')
    return s3
