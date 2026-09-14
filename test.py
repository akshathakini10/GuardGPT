from core.dataset_loader import DatasetLoader

loader = DatasetLoader()
loader.clear_cache()  # Deletes cache/guardgpt_faiss.index and cache/guardgpt_records.json
loader.load()        # Rebuilds the full index for all records