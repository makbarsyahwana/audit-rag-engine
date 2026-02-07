"""Cypher query templates for graph operations."""

# Upsert entity node
UPSERT_ENTITY = """
MERGE (e:Entity {id: $entity_id})
SET e.name = $name,
    e.type = $type,
    e.engagement_id = $engagement_id,
    e.description = $description,
    e.source_chunk_ids = $source_chunk_ids,
    e.source_document_ids = $source_document_ids,
    e.extraction_model = $extraction_model,
    e.schema_version = $schema_version,
    e.confidentiality_level = $confidentiality_level,
    e.embedding = $embedding
"""

# Upsert relationship between entities
UPSERT_RELATIONSHIP = """
MATCH (from_e:Entity {id: $from_entity_id})
MATCH (to_e:Entity {id: $to_entity_id})
MERGE (from_e)-[r:RELATES_TO {id: $relationship_id}]->(to_e)
SET r.type = $type,
    r.confidence = $confidence,
    r.source_chunk_id = $source_chunk_id,
    r.source_document_id = $source_document_id,
    r.engagement_id = $engagement_id
"""

# Create MENTIONED_IN links from entity to chunks
LINK_ENTITY_TO_CHUNKS = """
MATCH (e:Entity {id: $entity_id})
UNWIND $chunk_ids AS cid
MATCH (c:Chunk {id: cid})
MERGE (e)-[r:MENTIONED_IN]->(c)
SET r.confidence = $confidence
"""

# Deduplicate entities within an engagement (same name + type)
DEDUPLICATE_ENTITIES = """
MATCH (e:Entity {engagement_id: $engagement_id})
WITH e.name AS name, e.type AS type, collect(e) AS dupes
WHERE size(dupes) > 1
WITH dupes
UNWIND dupes[1..] AS dup
WITH dupes[0] AS keeper, dup
CALL {
    WITH keeper, dup
    MATCH (dup)-[:MENTIONED_IN]->(c:Chunk)
    MERGE (keeper)-[:MENTIONED_IN]->(c)
}
CALL {
    WITH keeper, dup
    SET keeper.source_chunk_ids = keeper.source_chunk_ids + dup.source_chunk_ids
}
DETACH DELETE dup
RETURN count(dup) AS merged
"""

# Find entities by name (fulltext)
SEARCH_ENTITIES_FULLTEXT = """
CALL db.index.fulltext.queryNodes('entity_fulltext', $search_term)
YIELD node AS entity, score
WHERE entity.engagement_id = $engagement_id
RETURN entity, score
ORDER BY score DESC
LIMIT $top_k
"""

# Traverse entity relationships
TRAVERSE_ENTITY_RELATIONSHIPS = """
MATCH (e:Entity {id: $entity_id})-[r:RELATES_TO*1..$depth]-(related:Entity)
RETURN e, r, related
LIMIT $limit
"""

# Get entity provenance (entity → chunks → documents)
ENTITY_PROVENANCE = """
MATCH (e:Entity {id: $entity_id})-[:MENTIONED_IN]->(c:Chunk)-[:BELONGS_TO]->(d:Document)
RETURN e, c, d
"""
