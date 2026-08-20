"""
Asynchronous Uploader for S3 and Azure Blob

Manages uploads of CSV files and screenshots to cloud storage.
Queue-based with worker threads for non-blocking uploads.
"""

import os
import queue
import threading
from pathlib import Path
from typing import Optional
from datetime import datetime


class S3Uploader:
    """S3 uploader."""
    
    def __init__(self, bucket: str, prefix: str = "", region: str = "us-east-1"):
        """
        Initialize S3 uploader.
        
        Args:
            bucket: S3 bucket name
            prefix: Optional prefix for object keys
            region: AWS region
        """
        self.bucket = bucket
        self.prefix = prefix.rstrip('/')
        self.region = region
        
        try:
            import boto3
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                region_name=region
            )
        except ImportError:
            print("Warning: boto3 not installed")
            self.s3_client = None
    
    def upload(self, local_path: str, file_type: str = "csv") -> bool:
        """
        Upload a file to S3.
        
        Args:
            local_path: Local file path
            file_type: File type ('csv' or 'screenshot')
            
        Returns:
            True if successful, False otherwise
        """
        if self.s3_client is None:
            return False
        
        try:
            path = Path(local_path)
            if not path.exists():
                return False
            
            # Generate S3 key
            date_str = datetime.now().strftime('%Y/%m/%d')
            filename = path.name
            if self.prefix:
                s3_key = f"{self.prefix}/{file_type}/{date_str}/{filename}"
            else:
                s3_key = f"{file_type}/{date_str}/{filename}"
            
            # Upload
            self.s3_client.upload_file(str(path), self.bucket, s3_key)
            print(f"Uploaded {local_path} to s3://{self.bucket}/{s3_key}")
            return True
        except Exception as e:
            print(f"Error uploading to S3: {e}")
            return False


class AzureBlobUploader:
    """Azure Blob Storage uploader."""
    
    def __init__(self, container: str, prefix: str = ""):
        """
        Initialize Azure Blob uploader.
        
        Args:
            container: Blob container name
            prefix: Optional prefix for blob names
        """
        self.container = container
        self.prefix = prefix.rstrip('/')
        
        try:
            from azure.storage.blob import BlobServiceClient
            connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
            if connection_string:
                self.blob_service = BlobServiceClient.from_connection_string(connection_string)
                self.container_client = self.blob_service.get_container_client(container)
            else:
                self.container_client = None
        except ImportError:
            print("Warning: azure-storage-blob not installed")
            self.container_client = None
    
    def upload(self, local_path: str, file_type: str = "csv") -> bool:
        """
        Upload a file to Azure Blob.
        
        Args:
            local_path: Local file path
            file_type: File type ('csv' or 'screenshot')
            
        Returns:
            True if successful, False otherwise
        """
        if self.container_client is None:
            return False
        
        try:
            path = Path(local_path)
            if not path.exists():
                return False
            
            # Generate blob name
            date_str = datetime.now().strftime('%Y/%m/%d')
            filename = path.name
            if self.prefix:
                blob_name = f"{self.prefix}/{file_type}/{date_str}/{filename}"
            else:
                blob_name = f"{file_type}/{date_str}/{filename}"
            
            # Upload
            with open(path, 'rb') as data:
                self.container_client.upload_blob(name=blob_name, data=data, overwrite=True)
            
            print(f"Uploaded {local_path} to Azure Blob: {blob_name}")
            return True
        except Exception as e:
            print(f"Error uploading to Azure Blob: {e}")
            return False


class UploadManager:
    """
    Asynchronous upload manager with queue-based workers.
    """
    
    def __init__(self, num_workers: int = 2):
        """
        Initialize upload manager.
        
        Args:
            num_workers: Number of worker threads
        """
        self.uploaders = []
        self.queue = queue.Queue(maxsize=1000)
        self.running = False
        self.workers = []
        
        self._init_uploaders()
        
        # Start worker threads
        self.running = True
        for i in range(num_workers):
            worker = threading.Thread(target=self._worker_loop, daemon=True, name=f"UploadWorker-{i}")
            worker.start()
            self.workers.append(worker)
    
    def _init_uploaders(self):
        """Initialize enabled uploaders based on environment variables."""
        # S3
        if os.getenv('S3_ENABLED', 'false').lower() == 'true':
            bucket = os.getenv('S3_BUCKET')
            prefix = os.getenv('S3_PREFIX', '')
            region = os.getenv('AWS_REGION', 'us-east-1')
            
            if bucket:
                uploader = S3Uploader(bucket, prefix, region)
                self.uploaders.append(uploader)
                print(f"Enabled S3 uploader (bucket: {bucket}, prefix: {prefix})")
        
        # Azure Blob
        if os.getenv('AZBLOB_ENABLED', 'false').lower() == 'true':
            container = os.getenv('AZURE_BLOB_CONTAINER')
            prefix = os.getenv('AZURE_BLOB_PREFIX', '')
            
            if container:
                uploader = AzureBlobUploader(container, prefix)
                self.uploaders.append(uploader)
                print(f"Enabled Azure Blob uploader (container: {container}, prefix: {prefix})")
    
    def queue_upload(self, local_path: str, file_type: str = "csv"):
        """
        Queue a file for upload.
        
        Args:
            local_path: Local file path
            file_type: File type ('csv' or 'screenshot')
        """
        if len(self.uploaders) == 0:
            return
        
        try:
            self.queue.put_nowait((local_path, file_type))
        except queue.Full:
            print(f"Warning: Upload queue full, dropping {local_path}")
    
    def _worker_loop(self):
        """Worker thread loop that processes upload queue."""
        while self.running:
            try:
                local_path, file_type = self.queue.get(timeout=1.0)
                
                # Upload to all enabled uploaders
                for uploader in self.uploaders:
                    uploader.upload(local_path, file_type)
                
                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in upload worker: {e}")
    
    def stop(self):
        """Stop upload manager and wait for workers."""
        self.running = False
        for worker in self.workers:
            worker.join(timeout=5.0)
    
    def get_queue_size(self) -> int:
        """Get current upload queue size."""
        return self.queue.qsize()


