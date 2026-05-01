import json
from retriever_config import build_elasticsearch_client, get_elasticsearch_index

class BM25Retriever:
    def __init__(self):
        self.es = build_elasticsearch_client(request_timeout=60)
        self.index = get_elasticsearch_index()

    def retrieve_docs(self, query: str, k: int = 10):
        es_query = {
            "size": k,
            "query": {
                "match": {
                    "content": query 
                }
            },
            "_source": ["PMID", "title", "content"]
        }
        # Execute the search query
        response = self.es.search(index=self.index, body=es_query)
        
        # Format the results into the desired JSON structure
        results = {}
        for idx, doc in enumerate(response['hits']['hits'], 1):
            doc_key = f"doc{idx}"
            results[doc_key] = {
                'PMID': doc['_source']['PMID'],
                'title': doc['_source']['title'],
                'content': doc['_source']['content'],
                'score': doc['_score']
            }

        return json.dumps(results, indent=4)