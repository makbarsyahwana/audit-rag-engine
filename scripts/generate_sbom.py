"""Generate Software Bill of Materials (SBOM) for the audit RAG engine (ASI04).

Produces a CycloneDX-compatible JSON SBOM from pyproject.toml dependencies.

Usage:
    python -m scripts.generate_sbom
"""

import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path


def parse_pyproject_deps(pyproject_path: Path) -> list[dict]:
    """Extract dependencies from pyproject.toml."""
    text = pyproject_path.read_text()

    # Find dependencies list
    deps_match = re.search(
        r'dependencies\s*=\s*\[(.*?)\]', text, re.DOTALL
    )
    if not deps_match:
        return []

    deps_block = deps_match.group(1)
    deps = []

    for line in deps_block.split("\n"):
        line = line.strip().strip('"').strip("'").strip(",")
        if not line or line.startswith("#"):
            continue

        # Parse "package>=version" or "package~=version"
        match = re.match(r'([a-zA-Z0-9_-]+)\s*([><=~!]+\s*[\d.]+)?', line)
        if match:
            name = match.group(1)
            version_spec = match.group(2) or ""
            deps.append({
                "type": "library",
                "name": name.lower(),
                "version": version_spec.strip(),
                "purl": f"pkg:pypi/{name.lower()}",
            })

    return deps


def generate_sbom(project_path: Path) -> dict:
    """Generate a CycloneDX-style SBOM."""
    pyproject = project_path / "pyproject.toml"
    if not pyproject.exists():
        print(f"Error: {pyproject} not found", file=sys.stderr)
        sys.exit(1)

    components = parse_pyproject_deps(pyproject)

    sbom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "version": 1,
        "metadata": {
            "timestamp": datetime.now(UTC).isoformat(),
            "component": {
                "type": "application",
                "name": "audit-rag-engine",
                "version": "0.1.0",
            },
            "tools": [{
                "name": "generate_sbom.py",
                "version": "1.0.0",
            }],
        },
        "components": components,
    }

    return sbom


def main() -> None:
    project_path = Path(__file__).parent.parent
    sbom = generate_sbom(project_path)

    output_path = project_path / "sbom.json"
    output_path.write_text(json.dumps(sbom, indent=2) + "\n")

    print(f"SBOM generated: {output_path}")
    print(f"  Components: {len(sbom['components'])}")
    print(f"  Timestamp: {sbom['metadata']['timestamp']}")


if __name__ == "__main__":
    main()
