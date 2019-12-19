import os

import boto3

# S3 Client 생성
from common.global_variables import PROJECT_HOME

s3 = boto3.client('s3')

# 업로드할 파일의 이름
filename = os.path.join(PROJECT_HOME, "codes", "tests", "aws", 'test.jpeg')

# 업로드할 S3 버킷
bucket_name = 'invest-thinkonweb'

# 첫본째 매개변수 : 로컬에서 올릴 파일이름
# 두번째 매개변수 : S3 버킷 이름
# 세번째 매개변수 : 버킷에 저장될 파일 이름.
s3.upload_file(filename, bucket_name, 'test.jpeg')
