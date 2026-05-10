"""
Blob Storage manager for Starlight catalogue page images.

Supports three backends — pick whichever suits your setup:

  LOCAL (default for localhost dev)
    Images saved to  AI-CRM-Mailer/static/page_images/
    URLs returned as http://localhost:8000/page_images/...
    Start the static server once with:
        python -m http.server 8000 --directory static
    Set in .env:  BLOB_BACKEND=local   (or leave blank — local is default)
    Optionally:   STATIC_SERVER_URL=http://localhost:8000  (default shown)

  CLOUDINARY (free tier — best for production emails with public URLs)
    25 GB storage + 25 GB bandwidth / month free.
    Sign up at https://cloudinary.com  → copy your CLOUDINARY_URL from Dashboard.
    Set in .env:  BLOB_BACKEND=cloudinary
                  CLOUDINARY_URL=cloudinary://API_KEY:API_SECRET@CLOUD_NAME
    Install:      pip install cloudinary

  AZURE (original — if you do have an Azure Storage account)
    Set in .env:  BLOB_BACKEND=azure
                  AZURE_STORAGE_CONNECTION_STRING=...
                  AZURE_STORAGE_CONTAINER=starlight-catalogues
    Install:      pip install azure-storage-blob
"""
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger("azure_blob")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BLOB_BACKEND = os.getenv("BLOB_BACKEND", "local").lower().strip()
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static", "page_images")
STATIC_SERVER_URL = os.getenv("STATIC_SERVER_URL", "http://localhost:8000").rstrip("/")
CONTAINER_DEFAULT = os.getenv("AZURE_STORAGE_CONTAINER", "starlight-catalogues")


# ===========================================================================
# Local file-system backend
# ===========================================================================

class _LocalBlobManager:
    """
    Saves page images to  static/page_images/<slug>/page_NNN.png
    Returns  http://localhost:8000/page_images/<slug>/page_NNN.png

    Start the companion server in a separate terminal:
        cd AI-CRM-Mailer
        python -m http.server 8000 --directory static
    """

    def upload_page_image(
        self,
        image_bytes: bytes,
        catalogue_slug: str,
        page_number: int,
        **_kwargs,
    ) -> str:
        folder = os.path.join(STATIC_DIR, catalogue_slug)
        os.makedirs(folder, exist_ok=True)

        filename = f"page_{page_number:03d}.png"
        filepath = os.path.join(folder, filename)
        with open(filepath, "wb") as f:
            f.write(image_bytes)

        url = f"{STATIC_SERVER_URL}/page_images/{catalogue_slug}/{filename}"
        log.debug("Saved locally: %s → %s", filepath, url)
        return url


# ===========================================================================
# Cloudinary backend (free tier — public URLs, no server needed)
# ===========================================================================

class _CloudinaryBlobManager:
    """
    Uploads page images to Cloudinary and returns permanent public HTTPS URLs.
    Free tier: 25 GB storage + 25 GB bandwidth / month.

    pip install cloudinary
    Set CLOUDINARY_URL=cloudinary://API_KEY:API_SECRET@CLOUD_NAME  in .env
    """

    def __init__(self) -> None:
        try:
            import cloudinary
            import cloudinary.uploader
            cloudinary.config(cloudinary_url=os.getenv("CLOUDINARY_URL", ""))
            self._cloudinary = cloudinary
        except ImportError:
            raise ImportError(
                "cloudinary package not installed. Run: pip install cloudinary"
            )

    def upload_page_image(
        self,
        image_bytes: bytes,
        catalogue_slug: str,
        page_number: int,
        **_kwargs,
    ) -> str:
        import io
        public_id = f"starlight/{catalogue_slug}/page_{page_number:03d}"
        result = self._cloudinary.uploader.upload(
            io.BytesIO(image_bytes),
            public_id=public_id,
            resource_type="image",
            format="png",
            overwrite=True,
        )
        url: str = result.get("secure_url", "")
        log.debug("Cloudinary upload: %s", url)
        return url


# ===========================================================================
# Azure Blob backend (original)
# ===========================================================================

class _AzureBlobManager:
    """Original Azure Blob Storage backend."""

    def __init__(self) -> None:
        try:
            from azure.storage.blob import BlobServiceClient, ContentSettings
            self._BlobServiceClient = BlobServiceClient
            self._ContentSettings = ContentSettings
        except ImportError:
            raise ImportError(
                "azure-storage-blob not installed. Run: pip install azure-storage-blob"
            )
        self.connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
        self.account_name = os.getenv("AZURE_STORAGE_ACCOUNT_NAME", "")
        self.account_key = os.getenv("AZURE_STORAGE_ACCOUNT_KEY", "")
        self.container = CONTAINER_DEFAULT
        self._client = None

    def _get_client(self):
        if self._client is None:
            if self.connection_string:
                self._client = self._BlobServiceClient.from_connection_string(
                    self.connection_string
                )
            else:
                url = f"https://{self.account_name}.blob.core.windows.net"
                self._client = self._BlobServiceClient(
                    account_url=url, credential=self.account_key
                )
        return self._client

    def _ensure_container(self):
        cc = self._get_client().get_container_client(self.container)
        try:
            cc.get_container_properties()
        except Exception:
            cc.create_container(public_access="blob")

    def upload_page_image(
        self,
        image_bytes: bytes,
        catalogue_slug: str,
        page_number: int,
        **_kwargs,
    ) -> str:
        self._ensure_container()
        blob_name = f"{catalogue_slug}/page_{page_number:03d}.png"
        blob = self._get_client().get_blob_client(container=self.container, blob=blob_name)
        blob.upload_blob(
            image_bytes,
            overwrite=True,
            content_settings=self._ContentSettings(content_type="image/png"),
        )
        for part in self.connection_string.split(";"):
            if part.startswith("AccountName="):
                acct = part.split("=", 1)[1]
                break
        else:
            acct = self.account_name
        return f"https://{acct}.blob.core.windows.net/{self.container}/{blob_name}"


# ===========================================================================
# Factory — returns the right manager based on BLOB_BACKEND
# ===========================================================================

def _make_manager():
    if BLOB_BACKEND == "cloudinary":
        log.info("Blob backend: Cloudinary")
        return _CloudinaryBlobManager()
    if BLOB_BACKEND == "azure":
        log.info("Blob backend: Azure Blob Storage")
        return _AzureBlobManager()
    # default
    log.info("Blob backend: local  (static/page_images/)")
    return _LocalBlobManager()


class AzureBlobManager:
    """
    Public façade — wraps the active backend.
    Import this class everywhere; swap backends via the BLOB_BACKEND env var.
    """

    def __init__(self) -> None:
        self._backend = None

    def _get_backend(self):
        if self._backend is None:
            self._backend = _make_manager()
        return self._backend

    @property
    def available(self) -> bool:
        return True  # local backend is always available

    def upload_page_image(
        self,
        image_bytes: bytes,
        catalogue_slug: str,
        page_number: int,
        content_type: str = "image/png",
    ) -> str:
        try:
            return self._get_backend().upload_page_image(
                image_bytes=image_bytes,
                catalogue_slug=catalogue_slug,
                page_number=page_number,
            )
        except Exception as exc:
            log.error("Blob upload failed (page %d): %s", page_number, exc)
            # Absolute fallback: save locally even if the chosen backend fails
            try:
                return _LocalBlobManager().upload_page_image(
                    image_bytes, catalogue_slug, page_number
                )
            except Exception:
                return f"[upload_failed]/{catalogue_slug}/page_{page_number:03d}.png"


# Module-level singleton
blob_manager = AzureBlobManager()
