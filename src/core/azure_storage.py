import json
from typing import Dict, List, Optional

from azure.storage.blob import BlobServiceClient, ContentSettings
from src.core.settings import Settings

class AzureStorage:
    def __init__(self):
        if not Settings.AZURE_CONNECTION_STRING:
            raise ValueError("AZURE_CONNECTION_STRING manquante")
        self.blob_service_client = BlobServiceClient.from_connection_string(
            Settings.AZURE_CONNECTION_STRING
        )
        self.container_client = self.blob_service_client.get_container_client(
            Settings.AZURE_CONTAINER
        )
        if not Settings.AZURE_CONNECTION_STRING:
            raise RuntimeError("AZURE_CONNECTION_STRING is missing. Set it in .env or environment variables.")


    def list_blobs(self, prefix: str) -> List[str]:
        blobs = self.container_client.list_blobs(name_starts_with=prefix)
        return sorted([b.name for b in blobs])

    def download_bytes(self, blob_name: str) -> Optional[bytes]:
        try:
            blob_client = self.container_client.get_blob_client(blob_name)
            return blob_client.download_blob().readall()
        except Exception:
            return None

    def download_json(self, blob_name: str) -> Optional[Dict]:
        raw = self.download_bytes(blob_name)
        if not raw:
            return None
        try:
            return json.loads(raw)
        except Exception:
            return None

    def upload_bytes(self, blob_name: str, data: bytes, content_type: str):
        blob_client = self.container_client.get_blob_client(blob_name)
        blob_client.upload_blob(
            data,
            overwrite=True,
            content_settings=ContentSettings(content_type=content_type),
        )

    def upload_json(self, blob_name: str, data: Dict):
        payload = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
        self.upload_bytes(blob_name, payload, "application/json")
