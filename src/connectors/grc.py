"""GRC (Governance, Risk, Compliance) system connector.

Supports generic REST-based GRC platforms (e.g., ServiceNow GRC, Archer,
MetricStream) via configurable endpoint mappings.
"""

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

# Default endpoint mappings — override via config.filters["endpoints"]
DEFAULT_ENDPOINTS = {
    "risks": "/api/v1/risks",
    "controls": "/api/v1/controls",
    "policies": "/api/v1/policies",
    "assessments": "/api/v1/assessments",
    "findings": "/api/v1/findings",
    "issues": "/api/v1/issues",
}

# GRC entity types that map to document types for ingestion
GRC_DOC_TYPE_MAP = {
    "risks": "risk_register",
    "controls": "control_document",
    "policies": "policy",
    "assessments": "assessment_report",
    "findings": "finding",
    "issues": "issue_report",
}


class GRCConnector(BaseConnector):
    """Connector for GRC platforms via REST API.

    Required credentials in config:
        api_key:   api_key (auth_type = "api_key")
        basic:     username + password (auth_type = "basic")
        oauth2:    client_id + client_secret + token_url (auth_type = "oauth2")

    Required filters:
        - entity_types: list of GRC entity types to sync
          (e.g., ["risks", "controls", "policies"])
        - endpoints (optional): dict overriding default API paths
        - status_filter (optional): e.g., "active" to filter records
    """

    connector_type = ConnectorType.GRC

    def __init__(self, config: ConnectorConfig) -> None:
        super().__init__(config)
        self._client: Optional[httpx.AsyncClient] = None
        self._access_token: Optional[str] = None

    async def connect(self) -> None:
        headers = self._build_headers()
        self._client = httpx.AsyncClient(
            base_url=self.config.base_url.rstrip("/"),
            headers=headers,
            timeout=30.0,
        )

        if self.config.auth_type == "oauth2":
            await self._authenticate_oauth2()

        self.logger.info("Connected to GRC system: %s", self.config.name)

    async def disconnect(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
        self.logger.info("Disconnected from GRC system: %s", self.config.name)

    async def test_connection(self) -> bool:
        try:
            endpoints = self._get_endpoints()
            first_endpoint = next(iter(endpoints.values()))
            resp = await self._client.get(  # type: ignore[union-attr]
                first_endpoint, params={"limit": 1}
            )
            return resp.status_code == 200
        except Exception as exc:
            self.logger.error("GRC connection test failed: %s", exc)
            return False

    async def list_documents(
        self,
        since: Optional[datetime] = None,
        **filters: Any,
    ) -> list[ConnectorDocument]:
        entity_types = (
            filters.get("entity_types")
            or self.config.filters.get("entity_types", list(DEFAULT_ENDPOINTS.keys()))
        )
        status_filter = (
            filters.get("status_filter")
            or self.config.filters.get("status_filter")
        )
        endpoints = self._get_endpoints()
        documents: list[ConnectorDocument] = []

        for entity_type in entity_types:
            endpoint = endpoints.get(entity_type)
            if not endpoint:
                self.logger.warning(
                    "No endpoint for entity type: %s", entity_type
                )
                continue

            params: dict[str, Any] = {"limit": 100, "offset": 0}
            if since:
                params["modified_after"] = since.isoformat()
            if status_filter:
                params["status"] = status_filter

            while True:
                resp = await self._client.get(  # type: ignore[union-attr]
                    endpoint, params=params
                )
                resp.raise_for_status()
                data = resp.json()

                items = data.get("data", data.get("results", []))
                if isinstance(items, dict):
                    items = [items]

                for item in items:
                    record_id = str(
                        item.get("id", item.get("sys_id", ""))
                    )
                    title = item.get(
                        "title",
                        item.get("name", item.get("number", record_id)),
                    )
                    modified_str = item.get(
                        "updated_at",
                        item.get("sys_updated_on", ""),
                    )
                    last_modified = None
                    if modified_str:
                        try:
                            last_modified = datetime.fromisoformat(
                                modified_str.replace("Z", "+00:00")
                            )
                        except ValueError:
                            pass

                    doc = ConnectorDocument(
                        external_id=f"{entity_type}:{record_id}",
                        title=title,
                        filename=f"{entity_type}_{record_id}.json",
                        content_type="application/json",
                        source_url=(
                            f"{self.config.base_url}"
                            f"{endpoint}/{record_id}"
                        ),
                        last_modified=last_modified,
                        metadata={
                            "entity_type": entity_type,
                            "doc_type": GRC_DOC_TYPE_MAP.get(
                                entity_type, "other"
                            ),
                            "status": item.get("status", ""),
                            "owner": item.get(
                                "owner",
                                item.get("assigned_to", ""),
                            ),
                            "risk_level": item.get("risk_level", ""),
                        },
                    )
                    documents.append(doc)

                # Pagination
                total = data.get("total", data.get("count", len(items)))
                params["offset"] += len(items)
                if params["offset"] >= total or len(items) == 0:
                    break

        self.logger.info(
            "Listed %d records from GRC system %s",
            len(documents),
            self.config.name,
        )
        return documents

    async def fetch_document(self, external_id: str) -> ConnectorDocument:
        entity_type, record_id = external_id.split(":", 1)
        endpoints = self._get_endpoints()
        endpoint = endpoints.get(entity_type, "")

        resp = await self._client.get(  # type: ignore[union-attr]
            f"{endpoint}/{record_id}"
        )
        resp.raise_for_status()
        data = resp.json()

        import json

        content = json.dumps(data, indent=2, default=str).encode("utf-8")

        title = data.get(
            "title", data.get("name", data.get("number", record_id))
        )

        return ConnectorDocument(
            external_id=external_id,
            title=title,
            filename=f"{entity_type}_{record_id}.json",
            content=content,
            content_type="application/json",
            source_url=f"{self.config.base_url}{endpoint}/{record_id}",
            metadata={
                "entity_type": entity_type,
                "doc_type": GRC_DOC_TYPE_MAP.get(entity_type, "other"),
            },
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_endpoints(self) -> dict[str, str]:
        custom = self.config.filters.get("endpoints", {})
        merged = {**DEFAULT_ENDPOINTS, **custom}
        return merged

    def _build_headers(self) -> dict[str, str]:
        creds = self.config.credentials
        headers: dict[str, str] = {"Accept": "application/json"}

        if self.config.auth_type == "api_key":
            api_key = creds.get("api_key", "")
            header_name = creds.get("header_name", "X-API-Key")
            headers[header_name] = api_key
        elif self.config.auth_type == "basic":
            import base64

            pair = f"{creds.get('username', '')}:{creds.get('password', '')}"
            encoded = base64.b64encode(pair.encode()).decode()
            headers["Authorization"] = f"Basic {encoded}"

        return headers

    async def _authenticate_oauth2(self) -> None:
        creds = self.config.credentials
        token_url = creds.get("token_url", "")
        if not token_url:
            self.logger.warning("OAuth2 token_url not provided for GRC connector")
            return

        resp = await self._client.post(  # type: ignore[union-attr]
            token_url,
            data={
                "grant_type": "client_credentials",
                "client_id": creds.get("client_id", ""),
                "client_secret": creds.get("client_secret", ""),
            },
        )
        resp.raise_for_status()
        data = resp.json()
        self._access_token = data["access_token"]

        # Update client headers
        self._client.headers["Authorization"] = (  # type: ignore[union-attr]
            f"Bearer {self._access_token}"
        )
        self.logger.debug("GRC OAuth2 token acquired")
