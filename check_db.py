import chromadb
import pprint

# Connect to the database
chroma_client = chromadb.PersistentClient(path="vector_db")
code_collection = chroma_client.get_collection(name="project_code")

# Get the total count of items
total_items = code_collection.count()
print(f"--- The 'project_code' collection contains {total_items} items. ---")

# Retrieve and print all items
all_items = code_collection.get()
print("\n--- All items in the collection: ---")
pprint.pprint(all_items)