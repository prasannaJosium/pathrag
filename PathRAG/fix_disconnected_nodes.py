import os
import logging
import asyncio
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fix_disconnected_nodes")

# Add the project root to sys.path
# Assuming this script is located at pathrag/PathRAG/fix_disconnected_nodes.py
# We want to add the parent directory of 'pathrag' (which is the project root) to sys.path
# However, based on the file structure, 'PathRAG' is a package inside 'pathrag'.
# Let's try to add the parent directory of the script to sys.path so we can import 'pathrag' as a module if needed,
# or if we are in the root, add the root.

# Current script location: D:\temp\contextgraph\pathrag\PathRAG\fix_disconnected_nodes.py
# We need to be able to import `from .config import PathRAGConfig` inside `PathRAG.py`.
# This implies running as a module or setting up sys.path correctly.

# Strategy: Add the directory containing 'pathrag' package to sys.path
# If script is at D:\temp\contextgraph\pathrag\PathRAG\fix_disconnected_nodes.py
# We want D:\temp\contextgraph to be in sys.path so we can do `from pathrag.PathRAG.PathRAG import PathRAG`

current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up two levels to get to D:\temp\contextgraph
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

# Now we should be able to import using full package paths
try:
    from pathrag.PathRAG.PathRAG import PathRAG
    from pathrag.PathRAG.config import PathRAGConfig
except ImportError:
    # Fallback: try adding just one level up if the structure is different
    sys.path.insert(0, os.path.dirname(current_dir))
    from PathRAG import PathRAG
    from config import PathRAGConfig


async def main():
    # Configure working directory - adjust based on where we expect data to be
    # Usually it's in project_root/pathrag/data
    working_dir = os.path.join(project_root, "pathrag", "data")

    if not os.path.exists(working_dir):
        # Fallback for different execution contexts
        working_dir = os.path.join(os.getcwd(), "pathrag", "data")

    if not os.path.exists(working_dir):
        logger.error(f"Working directory not found: {working_dir}")
        return

    logger.info(f"Initializing PathRAG with working directory: {working_dir}")

    # Initialize PathRAG
    config = PathRAGConfig(working_dir=working_dir)
    rag = PathRAG(config=config)

    logger.info("Starting disconnected node fix process...")
    logger.info("This may take a while depending on the number of disconnected nodes.")

    # Run the fix
    await rag.aretry_isolated_nodes()

    logger.info("Fix process completed.")


if __name__ == "__main__":
    asyncio.run(main())
