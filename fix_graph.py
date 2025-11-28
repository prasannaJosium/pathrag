import sys
import os
import asyncio
from dotenv import load_dotenv

# Ensure we can import from the current package
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Load .env from current_dir (pathrag/) BEFORE changing dir
env_path = os.path.join(current_dir, '.env')
if os.path.exists(env_path):
    print(f"Loading .env from {env_path}")
    load_dotenv(env_path)
else:
    # Try parent dir
    env_path = os.path.join(parent_dir, '.env')
    if os.path.exists(env_path):
        print(f"Loading .env from {env_path}")
        load_dotenv(env_path)
    else:
        print("Warning: .env file not found")

# Change CWD to parent dir (root of workspace) so that rag_manager uses correct data dir
print(f"Changing CWD to {parent_dir}")
os.chdir(parent_dir)

from pathrag.api.features.rag_manager import get_rag_instance
from pathrag.PathRAG.utils import logger

async def main():
    print("Starting graph fix process...")
    rag = get_rag_instance()
    
    print("Calling retry_linking...")
    try:
        # We access the async method directly
        await rag.aretry_linking()
        print("Graph fix process completed successfully.")
    except Exception as e:
        print(f"Error during graph fix: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
