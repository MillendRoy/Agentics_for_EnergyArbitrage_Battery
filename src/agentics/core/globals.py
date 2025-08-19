import os

from dotenv import load_dotenv
from memory_backend_client import Client
from memory_backend_client.api.default import (
    get_memory_dbs_list_get_memory_dbs_list_post,
    import_corpus_import_corpus_post,
    memorize_memorize_post,
    recall_recall_post,
)

load_dotenv()

if not os.path.exists(os.environ["AGENT_PERSISTANCE_PATH"]):
    os.mkdir(os.path.join(os.environ["AGENT_PERSISTANCE_PATH"]))


class Memory:
    memory_backend_url: str = "http://0.0.0.0:7816/"
    memory_client: Client

    async def get_collections(self):
        memory_client = Client(self.memory_backend_url)
        return await get_memory_dbs_list_get_memory_dbs_list_post.asyncio(
            client=memory_client
        )

    # def initialize_collection(self, collection_name:str):
    #     return initialize_memory_db_initialize_memory_db_post.sync(client=self.memory_client,memory_id=collection_name)
    async def import_corpus(self, collection_name: str, corpus_path: str) -> str:
        """
        If the corpus path is a file, ingest it and add to the specified collection.
        If it is a folder, recursively inspect it looking for documents of readable format.
        If the collection doesn't exist it creates a new one, otherwise update add to the existing one.
        """
        memory_client = Client(self.memory_backend_url)
        return await import_corpus_import_corpus_post.asyncio(
            client=memory_client,
            corpus_path=corpus_path,
            data_format="FOLDER",
            memory_id=collection_name,
        )

    async def memorize_corpus(
        self, collection_name: str, raw_text: str, id: str
    ) -> str:
        """
        If the corpus path is a file, ingest it and add to the specified collection.
        If it is a folder, recursively inspect it looking for documents of readable format.
        If the collection doesn't exist it creates a new one, otherwise update add to the existing one.
        """
        memory_client = Client(self.memory_backend_url)
        return await memorize_memorize_post.asyncio(
            client=memory_client,
            raw_text=raw_text,
            memory_id=collection_name,
            url_query=self.memory_backend_url,
            id=id,
        )

    def retrieve_content(self, collection_name: str, query, k: int = 5):
        """
        Retrieve the k most relevant passage of text for the given query from the specified collection
        """

        memory_client = Client(self.memory_backend_url)
        return recall_recall_post.sync(
            client=memory_client, memory_id=collection_name, query=query, k=k
        )

    def __init__(self, memory_backend_url: str = "http://0.0.0.0:7816/"):
        # self.memory_client=Client(memory_backend_url)
        pass
