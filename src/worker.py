"""Standalone worker process for async ingestion pipeline.

Connects to RabbitMQ, Neo4j, MongoDB, S3 and spawns consumers
for each pipeline stage queue. Can be scaled horizontally.

Usage:
    python -m src.worker
"""

import asyncio
import logging
import signal
import sys

from src.ingestion.stages import extract_handler, postprocess_handler, process_handler
from src.ingestion.task_broker import (
    QUEUE_EXTRACT,
    QUEUE_POSTPROCESS,
    QUEUE_PROCESS,
    task_broker,
)
from src.stores.document_store import document_store
from src.stores.neo4j_store import neo4j_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_shutdown_event = asyncio.Event()


def _handle_signal(sig, frame):
    logger.info("Received signal %s, shutting down...", sig)
    _shutdown_event.set()


async def run_worker() -> None:
    """Start the worker: connect to infrastructure and consume queues."""
    logger.info("Starting ingestion worker...")

    # Connect to infrastructure
    await neo4j_store.connect()
    await neo4j_store.ensure_indexes()
    await document_store.connect()
    await task_broker.connect()

    logger.info("Worker connected to all services")

    # Start consumers as concurrent tasks
    tasks = [
        asyncio.create_task(
            task_broker.consume(QUEUE_PROCESS, process_handler),
            name="process-consumer",
        ),
        asyncio.create_task(
            task_broker.consume(QUEUE_EXTRACT, extract_handler),
            name="extract-consumer",
        ),
        asyncio.create_task(
            task_broker.consume(QUEUE_POSTPROCESS, postprocess_handler),
            name="postprocess-consumer",
        ),
    ]

    logger.info(
        "Worker consuming from queues: %s, %s, %s",
        QUEUE_PROCESS, QUEUE_EXTRACT, QUEUE_POSTPROCESS,
    )

    # Wait for shutdown signal
    await _shutdown_event.wait()

    # Cancel consumer tasks
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    # Cleanup
    await task_broker.close()
    await neo4j_store.close()
    await document_store.close()
    logger.info("Worker shut down cleanly")


def main():
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        asyncio.run(run_worker())
    except KeyboardInterrupt:
        logger.info("Worker interrupted")
        sys.exit(0)


if __name__ == "__main__":
    main()
