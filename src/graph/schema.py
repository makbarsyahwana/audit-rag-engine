"""Graph schema definition for audit-domain entity types and relationship types."""

# Audit-domain entity types for ER extraction
ENTITY_TYPES = [
    "control",
    "requirement",
    "system",
    "process",
    "person",
    "organization",
    "document_ref",
    "risk",
    "policy",
    "framework",
]

# Audit-domain relationship types for ER extraction
RELATIONSHIP_TYPES = [
    "implements",
    "maps_to",
    "owns",
    "supports",
    "references",
    "depends_on",
    "part_of",
    "mitigates",
    "related_to",
]

# Schema description for LLM extraction prompt
GRAPH_SCHEMA_DESCRIPTION = """
Entity Types:
- control: An internal control (e.g., CHG-01, AC-02). Has an ID, description, owner, frequency.
- requirement: A framework clause or regulatory obligation (e.g., ISO 27001 A.12.1, GDPR Art.32).
- system: An IT system, application, or platform (e.g., SAP, Active Directory, AWS).
- process: A business or IT process (e.g., change management, access provisioning).
- person: A named individual mentioned in the document (e.g., control owner, auditor).
- organization: A company, department, or business unit.
- document_ref: A reference to another document (e.g., policy name, standard title).
- risk: A risk or threat (e.g., unauthorized access, data loss).
- policy: An internal policy document.
- framework: An external standard or framework (e.g., ISO 27001, NIST CSF, COBIT).

Relationship Types:
- implements: A control implements a requirement.
- maps_to: A requirement maps to a control or another requirement.
- owns: A person or organization owns a control or process.
- supports: A system supports a process or control.
- references: A document references another entity.
- depends_on: An entity depends on another entity.
- part_of: An entity is part of another entity.
- mitigates: A control mitigates a risk.
- related_to: General relationship when a more specific type doesn't apply.
"""
