"""Atlassian Confluence connector using REST API v2."""

import logging
from datetime import datetime
from typing import Any, Optional

import httpx

from src.connectors.base import (
    BaseConnector,
    ConnectorConfig,
    ConnectorDocument,
    ConnectorType,
)

logger = logging.getLogger(__name__)


class ConfluenceConnector(BaseConnector):
    """Connector for Atlassian Confluence (Cloud or Data Center).

    Required credentials in config:
        Cloud:  email + api_token  (auth_type = "api_key")
        DC:     username + password (auth_type = "basic") or pat (auth_type = "pat")

    Required filters:
        - space_key: Confluence space to sync
        - content_type (optional): "page" | "blogpost" (default: "page")
        - ancestor_id (optional): Only sync descendants of this page
        - labels (optional): list of labels to filter on
    """

    connector_type = ConnectorType.CONFLUENCE

    def __init__(self, config: ConnectorConfig) -> None:
        super().__init__(config)
        self._client: Optional[httpx.AsyncClient] = None

    async def connect(self) -> None:
        auth = self._build_auth()
        self._client = httpx.AsyncClient(
            base_url=self.config.base_url.rstrip("/"),
            auth=auth,
            timeout=30.0,
        )
        self.logger.info("Connected to Confluence: %s", self.config.name)

    async def disconnect(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
        self.logger.info("Disconnected from Confluence: %s", self.config.name)

    async def test_connection(self) -> bool:
        try:
            resp = await self._client.get("/wiki/api/v2/spaces", params={"limit": 1})  # type: ignore[union-attr]
            return resp.status_code == 200
        except Exception as exc:
            self.logger.error("Confluence connection test failed: %s", exc)
            return False

    async def list_documents(
        self,
        since: Optional[datetime] = None,
        **filters: Any,
    ) -> list[ConnectorDocument]:
        space_key = filters.get("space_key") or self.config.filters.get("space_key", "")
        content_type = filters.get("content_type") or self.config.filters.get(
            "content_type", "page"
        )
        labels = filters.get("labels") or self.config.filters.get("labels", [])
        ancestor_id = filters.get("ancestor_id") or self.config.filters.get("ancestor_id")

        documents: list[ConnectorDocument] = []

        # Use CQL for flexible querying
        cql_parts = [f'space = "{space_key}"', f'type = "{content_type}"']
        if ancestor_id:
            cql_parts.append(f'ancestor = "{ancestor_id}"')
        if labels:
            label_str = ",".join(f'"{lbl}"' for lbl in labels)
            cql_parts.append(f"label in ({label_str})")
        if since:
            cql_parts.append(f'lastModified > "{since.strftime("%Y-%m-%d")}"')

        cql = " AND ".join(cql_parts)
        cql += " ORDER BY lastModified DESC"

        start = 0
        limit = 50

        while True:
            resp = await self._client.get(  # type: ignore[union-attr]
                "/wiki/rest/api/content/search",
                params={
                    "cql": cql,
                    "start": start,
                    "limit": limit,
                    "expand": "version,history.lastUpdated",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])

            for item in results:
                last_updated_str = (
                    item.get("history", {})
                    .get("lastUpdated", {})
                    .get("when", "")
                )
                last_modified = None
                if last_updated_str:
                    try:
                        last_modified = datetime.fromisoformat(
                            last_updated_str.replace("Z", "+00:00")
                        )
                    except ValueError:
                        pass

                web_path = item.get("_links", {}).get("webui", "")
                doc = ConnectorDocument(
                    external_id=item["id"],
                    title=item.get("title", ""),
                    filename=f"{item.get('title', 'untitled')}.html",
                    content_type="text/html",
                    source_url=f"{self.config.base_url}/wiki{web_path}",
                    last_modified=last_modified,
                    metadata={
                        "space_key": space_key,
                        "content_type": item.get("type", content_type),
                        "version": item.get("version", {}).get("number", 1),
                        "status": item.get("status", "current"),
                        "labels": [
                            lbl["name"]
                            for lbl in item.get("metadata", {})
                            .get("labels", {})
                            .get("results", [])
                        ],
                    },
                )
                documents.append(doc)

            # Pagination
            if len(results) < limit:
                break
            start += limit

        self.logger.info(
            "Listed %d documents from Confluence space %s",
            len(documents),
            space_key,
        )
        return documents

    async def fetch_document(self, external_id: str) -> ConnectorDocument:
        # Get page metadata + body
        resp = await self._client.get(  # type: ignore[union-attr]
            f"/wiki/rest/api/content/{external_id}",
            params={"expand": "body.export_view,version,history.lastUpdated"},
        )
        resp.raise_for_status()
        item = resp.json()

        body_html = (
            item.get("body", {}).get("export_view", {}).get("value", "")
        )
        content = body_html.encode("utf-8")

        last_updated_str = (
            item.get("history", {}).get("lastUpdated", {}).get("when", "")
        )
        last_modified = None
        if last_updated_str:
            try:
                last_modified = datetime.fromisoformat(
                    last_updated_str.replace("Z", "+00:00")
                )
            except ValueError:
                pass

        return ConnectorDocument(
            external_id=external_id,
            title=item.get("title", ""),
            filename=f"{item.get('title', 'untitled')}.html",
            content=content,
            content_type="text/html",
            source_url=f"{self.config.base_url}/wiki{item.get('_links', {}).get('webui', '')}",
            last_modified=last_modified,
            metadata={
                "space_key": item.get("space", {}).get("key", ""),
                "version": item.get("version", {}).get("number", 1),
            },
        )

    # Also support fetching attachments from a page
    async def list_attachments(self, page_id: str) -> list[ConnectorDocument]:
        """List file attachments on a Confluence page."""
        documents: list[ConnectorDocument] = []
        resp = await self._client.get(  # type: ignore[union-attr]
            f"/wiki/rest/api/content/{page_id}/child/attachment",
            params={"expand": "version"},
        )
        resp.raise_for_status()
        data = resp.json()

        for item in data.get("results", []):
            download_link = item.get("_links", {}).get("download", "")
            doc = ConnectorDocument(
                external_id=item["id"],
                title=item.get("title", ""),
                filename=item.get("title", ""),
                content_type=item.get("metadata", {}).get(
                    "mediaType", "application/octet-stream"
                ),
                source_url=f"{self.config.base_url}/wiki{download_link}",
                metadata={
                    "page_id": page_id,
                    "file_size": item.get("extensions", {}).get("fileSize", 0),
                    "version": item.get("version", {}).get("number", 1),
                },
            )
            documents.append(doc)
        return documents

    async def fetch_attachment(self, attachment_id: str) -> ConnectorDocument:
        """Download an attachment's content."""
        resp = await self._client.get(  # type: ignore[union-attr]
            f"/wiki/rest/api/content/{attachment_id}",
            params={"expand": "version"},
        )
        resp.raise_for_status()
        item = resp.json()

        download_link = item.get("_links", {}).get("download", "")
        content_resp = await self._client.get(  # type: ignore[union-attr]
            f"/wiki{download_link}",
        )
        content_resp.raise_for_status()

        return ConnectorDocument(
            external_id=attachment_id,
            title=item.get("title", ""),
            filename=item.get("title", ""),
            content=content_resp.content,
            content_type=item.get("metadata", {}).get(
                "mediaType", "application/octet-stream"
            ),
            source_url=f"{self.config.base_url}/wiki{download_link}",
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_auth(self) -> Optional[httpx.BasicAuth]:
        creds = self.config.credentials
        auth_type = self.config.auth_type

        if auth_type == "api_key":
            # Confluence Cloud: email + API token
            return httpx.BasicAuth(
                username=creds.get("email", ""),
                password=creds.get("api_token", ""),
            )
        elif auth_type == "basic":
            return httpx.BasicAuth(
                username=creds.get("username", ""),
                password=creds.get("password", ""),
            )
        elif auth_type == "pat":
            # Personal Access Token - use as Bearer token
            # httpx doesn't directly support bearer, so we use basic with empty user
            return httpx.BasicAuth(
                username="",
                password=creds.get("pat", ""),
            )

        return None
