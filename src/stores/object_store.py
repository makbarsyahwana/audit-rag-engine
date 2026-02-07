"""S3/MinIO client for raw file storage."""

import logging
from io import BytesIO
from typing import Optional

import boto3
from botocore.exceptions import ClientError

from src.config import settings

logger = logging.getLogger(__name__)


class ObjectStore:
    """S3-compatible object storage client for raw audit documents."""

    _client: Optional[object] = None

    def connect(self) -> None:
        """Initialize the S3/MinIO client."""
        self._client = boto3.client(
            "s3",
            endpoint_url=settings.s3_endpoint,
            aws_access_key_id=settings.s3_access_key,
            aws_secret_access_key=settings.s3_secret_key,
        )
        self._ensure_bucket()
        logger.info("Connected to S3/MinIO at %s", settings.s3_endpoint)

    def _ensure_bucket(self) -> None:
        """Create the bucket if it doesn't exist."""
        try:
            self.client.head_bucket(Bucket=settings.s3_bucket)
        except ClientError:
            self.client.create_bucket(Bucket=settings.s3_bucket)
            logger.info("Created S3 bucket: %s", settings.s3_bucket)

    @property
    def client(self):
        if not self._client:
            raise RuntimeError("S3 client not initialized. Call connect() first.")
        return self._client

    def upload_file(
        self, key: str, data: bytes, content_type: str = "application/octet-stream"
    ) -> str:
        """Upload a file to S3. Returns the object key."""
        self.client.put_object(
            Bucket=settings.s3_bucket,
            Key=key,
            Body=data,
            ContentType=content_type,
        )
        logger.info("Uploaded %s to S3 bucket %s", key, settings.s3_bucket)
        return key

    def download_file(self, key: str) -> bytes:
        """Download a file from S3."""
        response = self.client.get_object(Bucket=settings.s3_bucket, Key=key)
        return response["Body"].read()

    def download_to_buffer(self, key: str) -> BytesIO:
        """Download a file from S3 into a BytesIO buffer."""
        data = self.download_file(key)
        return BytesIO(data)

    def delete_file(self, key: str) -> None:
        """Delete a file from S3."""
        self.client.delete_object(Bucket=settings.s3_bucket, Key=key)
        logger.info("Deleted %s from S3 bucket %s", key, settings.s3_bucket)

    def get_presigned_url(self, key: str, expires_in: int = 3600) -> str:
        """Generate a presigned URL for temporary access."""
        return self.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": settings.s3_bucket, "Key": key},
            ExpiresIn=expires_in,
        )


object_store = ObjectStore()
