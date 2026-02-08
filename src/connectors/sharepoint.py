"""SharePoint Online connector using Microsoft Graph API."""

import logging
from datetime import UTC, datetime
from typing import Any, Optional

import httpx

from src.connectors.base import (
    BaseConnector,
    ConnectorConfig,
    ConnectorDocument,
    ConnectorType,
)

logger = logging.getLogger(__name__)

# Microsoft Graph API endpoints
GRAPH_BASE = "https://graph.microsoft.com/v1.0"
TOKEN_URL = "https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"


class SharePointConnector(BaseConnector):
    """Connector for SharePoint Online via Microsoft Graph API.

    Required credentials in config:
        - tenant_id: Azure AD tenant ID
        - client_id: App registration client ID
        - client_secret: App registration client secret

    Required filters:
        - site_id or site_url: Target SharePoint site
        - drive_id (optional): Specific document library
        - folder_path (optional): Subfolder to sync
    """

    connector_type = ConnectorType.SHAREPOINT

    def __init__(self, config: ConnectorConfig) -> None:
        super().__init__(config)
        self._client: Optional[httpx.AsyncClient] = None
        self._access_token: Optional[str] = None
        self._token_expires_at: Optional[datetime] = None

    async def connect(self) -> None:
        self._client = httpx.AsyncClient(timeout=30.0)
        await self._refresh_token()
        self.logger.info("Connected to SharePoint: %s", self.config.name)

    async def disconnect(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
        self.logger.info("Disconnected from SharePoint: %s", self.config.name)

    async def test_connection(self) -> bool:
        try:
            await self._ensure_token()
            site_id = await self._resolve_site_id()
            resp = await self._graph_get(f"/sites/{site_id}")
            return resp.status_code == 200
        except Exception as exc:
            self.logger.error("SharePoint connection test failed: %s", exc)
            return False

    async def list_documents(
        self,
        since: Optional[datetime] = None,
        **filters: Any,
    ) -> list[ConnectorDocument]:
        await self._ensure_token()
        site_id = await self._resolve_site_id()
        drive_id = self.config.filters.get("drive_id") or filters.get("drive_id")

        # If no drive_id, use the default document library
        if not drive_id:
            resp = await self._graph_get(f"/sites/{site_id}/drive")
            resp.raise_for_status()
            drive_id = resp.json()["id"]

        # Build the endpoint
        folder_path = self.config.filters.get("folder_path", "")
        if folder_path:
            endpoint = f"/drives/{drive_id}/root:/{folder_path.strip('/')}:/children"
        else:
            endpoint = f"/drives/{drive_id}/root/children"

        documents: list[ConnectorDocument] = []
        next_link: Optional[str] = endpoint

        while next_link:
            if next_link.startswith("http"):
                resp = await self._client.get(  # type: ignore[union-attr]
                    next_link,
                    headers=self._auth_headers(),
                )
            else:
                resp = await self._graph_get(next_link)

            resp.raise_for_status()
            data = resp.json()

            for item in data.get("value", []):
                # Skip folders
                if "folder" in item:
                    continue

                last_modified = datetime.fromisoformat(
                    item["lastModifiedDateTime"].replace("Z", "+00:00")
                )

                # Incremental sync: skip if not modified since last sync
                if since and last_modified <= since:
                    continue

                doc = ConnectorDocument(
                    external_id=item["id"],
                    title=item.get("name", ""),
                    filename=item.get("name", ""),
                    content_type=item.get("file", {}).get(
                        "mimeType", "application/octet-stream"
                    ),
                    source_url=item.get("webUrl", ""),
                    last_modified=last_modified,
                    metadata={
                        "drive_id": drive_id,
                        "size": item.get("size", 0),
                        "created_by": item.get("createdBy", {})
                        .get("user", {})
                        .get("displayName", ""),
                        "modified_by": item.get("lastModifiedBy", {})
                        .get("user", {})
                        .get("displayName", ""),
                        "parent_path": item.get("parentReference", {}).get(
                            "path", ""
                        ),
                    },
                )
                documents.append(doc)

            next_link = data.get("@odata.nextLink")

        self.logger.info(
            "Listed %d documents from SharePoint site %s",
            len(documents),
            site_id,
        )
        return documents

    async def fetch_document(self, external_id: str) -> ConnectorDocument:
        await self._ensure_token()
        site_id = await self._resolve_site_id()
        drive_id = self.config.filters.get("drive_id")

        if not drive_id:
            resp = await self._graph_get(f"/sites/{site_id}/drive")
            resp.raise_for_status()
            drive_id = resp.json()["id"]

        # Get item metadata
        meta_resp = await self._graph_get(f"/drives/{drive_id}/items/{external_id}")
        meta_resp.raise_for_status()
        item = meta_resp.json()

        # Download content
        content_resp = await self._graph_get(
            f"/drives/{drive_id}/items/{external_id}/content",
        )
        content_resp.raise_for_status()

        return ConnectorDocument(
            external_id=external_id,
            title=item.get("name", ""),
            filename=item.get("name", ""),
            content=content_resp.content,
            content_type=item.get("file", {}).get(
                "mimeType", "application/octet-stream"
            ),
            source_url=item.get("webUrl", ""),
            last_modified=datetime.fromisoformat(
                item["lastModifiedDateTime"].replace("Z", "+00:00")
            ),
            metadata={
                "drive_id": drive_id,
                "size": item.get("size", 0),
            },
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _refresh_token(self) -> None:
        """Obtain or refresh the OAuth2 access token."""
        creds = self.config.credentials
        tenant_id = creds.get("tenant_id", "")
        url = TOKEN_URL.format(tenant_id=tenant_id)

        resp = await self._client.post(  # type: ignore[union-attr]
            url,
            data={
                "grant_type": "client_credentials",
                "client_id": creds.get("client_id", ""),
                "client_secret": creds.get("client_secret", ""),
                "scope": "https://graph.microsoft.com/.default",
            },
        )
        resp.raise_for_status()
        data = resp.json()

        self._access_token = data["access_token"]
        expires_in = data.get("expires_in", 3600)
        self._token_expires_at = datetime.now(UTC).__class__.now(UTC).__class__(
            *datetime.now(UTC).timetuple()[:6]
        )
        # Simplified: refresh 5 min before expiry
        from datetime import timedelta

        self._token_expires_at = datetime.now(UTC) + timedelta(
            seconds=expires_in - 300
        )
        self.logger.debug("SharePoint token refreshed, expires in %ds", expires_in)

    async def _ensure_token(self) -> None:
        """Refresh token if expired."""
        if (
            self._access_token is None
            or self._token_expires_at is None
            or datetime.now(UTC) >= self._token_expires_at
        ):
            await self._refresh_token()

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._access_token}"}

    async def _graph_get(self, path: str) -> httpx.Response:
        """Make an authenticated GET to Microsoft Graph."""
        url = f"{GRAPH_BASE}{path}" if not path.startswith("http") else path
        return await self._client.get(url, headers=self._auth_headers())  # type: ignore[union-attr]

    async def _resolve_site_id(self) -> str:
        """Resolve site_id from config (direct ID or URL-based lookup)."""
        site_id = self.config.filters.get("site_id")
        if site_id:
            return site_id

        site_url = self.config.filters.get("site_url", "")
        if site_url:
            # Parse hostname and site path from URL
            from urllib.parse import urlparse

            parsed = urlparse(site_url)
            hostname = parsed.hostname or ""
            site_path = parsed.path.rstrip("/")
            resp = await self._graph_get(
                f"/sites/{hostname}:{site_path}"
            )
            resp.raise_for_status()
            return resp.json()["id"]

        raise ValueError(
            "SharePoint connector requires 'site_id' or 'site_url' in filters."
        )
