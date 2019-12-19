import os

import boto3

# S3 Client 생성
from common.global_variables import PROJECT_HOME

# 업로드할 파일의 이름
filename = os.path.join(PROJECT_HOME, "codes", "tests", "aws", 'test.jpeg')

# 업로드할 S3 버킷
bucket_name = 'invest-thinkonweb'

s3 = boto3.client('s3')
s3.download_file(bucket_name, 'test1.jpeg', filename)
