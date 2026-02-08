"""Ticketing system connectors for Jira and ServiceNow."""

import json
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


# ---------------------------------------------------------------------------
# Jira Connector
# ---------------------------------------------------------------------------


class JiraConnector(BaseConnector):
    """Connector for Atlassian Jira (Cloud or Data Center).

    Required credentials in config:
        Cloud:  email + api_token  (auth_type = "api_key")
        DC:     username + password (auth_type = "basic")

    Required filters:
        - project_key: Jira project key (e.g., "AUD")
        - jql (optional): Custom JQL query override
        - issue_types (optional): list (e.g., ["Bug", "Task", "Story"])
        - statuses (optional): list (e.g., ["Open", "In Progress"])
    """

    connector_type = ConnectorType.JIRA

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
        self.logger.info("Connected to Jira: %s", self.config.name)

    async def disconnect(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def test_connection(self) -> bool:
        try:
            resp = await self._client.get(  # type: ignore[union-attr]
                "/rest/api/3/myself"
            )
            return resp.status_code == 200
        except Exception as exc:
            self.logger.error("Jira connection test failed: %s", exc)
            return False

    async def list_documents(
        self,
        since: Optional[datetime] = None,
        **filters: Any,
    ) -> list[ConnectorDocument]:
        jql = self._build_jql(since, **filters)
        documents: list[ConnectorDocument] = []
        start_at = 0
        max_results = 50

        while True:
            resp = await self._client.get(  # type: ignore[union-attr]
                "/rest/api/3/search",
                params={
                    "jql": jql,
                    "startAt": start_at,
                    "maxResults": max_results,
                    "fields": "summary,status,issuetype,updated,"
                    "assignee,reporter,priority,labels",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            issues = data.get("issues", [])

            for issue in issues:
                fields = issue.get("fields", {})
                updated_str = fields.get("updated", "")
                last_modified = None
                if updated_str:
                    try:
                        last_modified = datetime.fromisoformat(
                            updated_str.replace("Z", "+00:00")
                        )
                    except ValueError:
                        pass

                doc = ConnectorDocument(
                    external_id=issue["key"],
                    title=f"{issue['key']}: {fields.get('summary', '')}",
                    filename=f"{issue['key']}.json",
                    content_type="application/json",
                    source_url=f"{self.config.base_url}/browse/{issue['key']}",
                    last_modified=last_modified,
                    metadata={
                        "issue_type": (
                            fields.get("issuetype", {}).get("name", "")
                        ),
                        "status": (
                            fields.get("status", {}).get("name", "")
                        ),
                        "priority": (
                            fields.get("priority", {}).get("name", "")
                        ),
                        "assignee": (
                            fields.get("assignee", {}) or {}
                        ).get("displayName", ""),
                        "labels": fields.get("labels", []),
                    },
                )
                documents.append(doc)

            total = data.get("total", 0)
            start_at += len(issues)
            if start_at >= total or len(issues) == 0:
                break

        self.logger.info(
            "Listed %d issues from Jira project", len(documents)
        )
        return documents

    async def fetch_document(self, external_id: str) -> ConnectorDocument:
        resp = await self._client.get(  # type: ignore[union-attr]
            f"/rest/api/3/issue/{external_id}",
            params={"expand": "renderedFields"},
        )
        resp.raise_for_status()
        data = resp.json()

        content = json.dumps(data, indent=2, default=str).encode("utf-8")
        fields = data.get("fields", {})

        return ConnectorDocument(
            external_id=external_id,
            title=f"{data['key']}: {fields.get('summary', '')}",
            filename=f"{data['key']}.json",
            content=content,
            content_type="application/json",
            source_url=f"{self.config.base_url}/browse/{data['key']}",
            last_modified=None,
            metadata={
                "issue_type": (
                    fields.get("issuetype", {}).get("name", "")
                ),
                "status": fields.get("status", {}).get("name", ""),
            },
        )

    def _build_jql(
        self, since: Optional[datetime] = None, **filters: Any
    ) -> str:
        custom_jql = (
            filters.get("jql") or self.config.filters.get("jql")
        )
        if custom_jql:
            return custom_jql

        project_key = (
            filters.get("project_key")
            or self.config.filters.get("project_key", "")
        )
        parts = [f'project = "{project_key}"']

        issue_types = (
            filters.get("issue_types")
            or self.config.filters.get("issue_types", [])
        )
        if issue_types:
            types_str = ",".join(f'"{t}"' for t in issue_types)
            parts.append(f"issuetype in ({types_str})")

        statuses = (
            filters.get("statuses")
            or self.config.filters.get("statuses", [])
        )
        if statuses:
            status_str = ",".join(f'"{s}"' for s in statuses)
            parts.append(f"status in ({status_str})")

        if since:
            parts.append(
                f'updated >= "{since.strftime("%Y-%m-%d %H:%M")}"'
            )

        return " AND ".join(parts) + " ORDER BY updated DESC"

    def _build_auth(self) -> Optional[httpx.BasicAuth]:
        creds = self.config.credentials
        if self.config.auth_type in ("api_key", "basic"):
            return httpx.BasicAuth(
                username=creds.get("email", creds.get("username", "")),
                password=creds.get(
                    "api_token", creds.get("password", "")
                ),
            )
        return None


# ---------------------------------------------------------------------------
# ServiceNow Connector
# ---------------------------------------------------------------------------


class ServiceNowConnector(BaseConnector):
    """Connector for ServiceNow via Table API.

    Required credentials in config:
        basic:  username + password (auth_type = "basic")
        oauth2: client_id + client_secret (auth_type = "oauth2")

    Required filters:
        - tables: list of ServiceNow tables to sync
          (e.g., ["incident", "change_request", "problem"])
        - query (optional): encoded query string per table
    """

    connector_type = ConnectorType.SERVICENOW

    def __init__(self, config: ConnectorConfig) -> None:
        super().__init__(config)
        self._client: Optional[httpx.AsyncClient] = None

    async def connect(self) -> None:
        headers: dict[str, str] = {"Accept": "application/json"}
        auth = None

        if self.config.auth_type == "basic":
            creds = self.config.credentials
            auth = httpx.BasicAuth(
                username=creds.get("username", ""),
                password=creds.get("password", ""),
            )

        self._client = httpx.AsyncClient(
            base_url=self.config.base_url.rstrip("/"),
            headers=headers,
            auth=auth,
            timeout=30.0,
        )

        if self.config.auth_type == "oauth2":
            await self._authenticate_oauth2()

        self.logger.info("Connected to ServiceNow: %s", self.config.name)

    async def disconnect(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def test_connection(self) -> bool:
        try:
            resp = await self._client.get(  # type: ignore[union-attr]
                "/api/now/table/sys_user",
                params={"sysparm_limit": 1},
            )
            return resp.status_code == 200
        except Exception as exc:
            self.logger.error("ServiceNow connection test failed: %s", exc)
            return False

    async def list_documents(
        self,
        since: Optional[datetime] = None,
        **filters: Any,
    ) -> list[ConnectorDocument]:
        tables = (
            filters.get("tables")
            or self.config.filters.get(
                "tables", ["incident", "change_request"]
            )
        )
        documents: list[ConnectorDocument] = []

        for table in tables:
            params: dict[str, Any] = {
                "sysparm_limit": 100,
                "sysparm_offset": 0,
                "sysparm_display_value": "true",
                "sysparm_fields": (
                    "sys_id,number,short_description,state,"
                    "priority,assigned_to,sys_updated_on"
                ),
            }

            query_parts = []
            custom_query = self.config.filters.get("query", {}).get(
                table, ""
            )
            if custom_query:
                query_parts.append(custom_query)
            if since:
                query_parts.append(
                    f"sys_updated_on>={since.strftime('%Y-%m-%d %H:%M:%S')}"
                )
            if query_parts:
                params["sysparm_query"] = "^".join(query_parts)

            while True:
                resp = await self._client.get(  # type: ignore[union-attr]
                    f"/api/now/table/{table}",
                    params=params,
                )
                resp.raise_for_status()
                data = resp.json()
                records = data.get("result", [])

                for record in records:
                    sys_id = record.get("sys_id", "")
                    number = record.get("number", sys_id)
                    updated_str = record.get("sys_updated_on", "")
                    last_modified = None
                    if updated_str:
                        try:
                            last_modified = datetime.fromisoformat(
                                updated_str
                            )
                        except ValueError:
                            pass

                    doc = ConnectorDocument(
                        external_id=f"{table}:{sys_id}",
                        title=(
                            f"{number}: "
                            f"{record.get('short_description', '')}"
                        ),
                        filename=f"{table}_{number}.json",
                        content_type="application/json",
                        source_url=(
                            f"{self.config.base_url}"
                            f"/nav_to.do?uri={table}.do?sys_id={sys_id}"
                        ),
                        last_modified=last_modified,
                        metadata={
                            "table": table,
                            "state": record.get("state", ""),
                            "priority": record.get("priority", ""),
                            "assigned_to": record.get(
                                "assigned_to", ""
                            ),
                        },
                    )
                    documents.append(doc)

                # Pagination
                if len(records) < 100:
                    break
                params["sysparm_offset"] += len(records)

        self.logger.info(
            "Listed %d records from ServiceNow", len(documents)
        )
        return documents

    async def fetch_document(self, external_id: str) -> ConnectorDocument:
        table, sys_id = external_id.split(":", 1)
        resp = await self._client.get(  # type: ignore[union-attr]
            f"/api/now/table/{table}/{sys_id}",
            params={"sysparm_display_value": "true"},
        )
        resp.raise_for_status()
        data = resp.json().get("result", {})

        content = json.dumps(data, indent=2, default=str).encode("utf-8")
        number = data.get("number", sys_id)

        return ConnectorDocument(
            external_id=external_id,
            title=f"{number}: {data.get('short_description', '')}",
            filename=f"{table}_{number}.json",
            content=content,
            content_type="application/json",
            source_url=(
                f"{self.config.base_url}"
                f"/nav_to.do?uri={table}.do?sys_id={sys_id}"
            ),
            metadata={"table": table},
        )

    async def _authenticate_oauth2(self) -> None:
        creds = self.config.credentials
        resp = await self._client.post(  # type: ignore[union-attr]
            "/oauth_token.do",
            data={
                "grant_type": "client_credentials",
                "client_id": creds.get("client_id", ""),
                "client_secret": creds.get("client_secret", ""),
            },
        )
        resp.raise_for_status()
        token = resp.json().get("access_token", "")
        self._client.headers["Authorization"] = f"Bearer {token}"  # type: ignore[union-attr]
        self.logger.debug("ServiceNow OAuth2 token acquired")
