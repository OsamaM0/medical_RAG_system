import os

from elasticsearch import Elasticsearch


def env_bool(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def first_env(*names):
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def build_elasticsearch_client(request_timeout=60):
    cloud_id = os.getenv("ELASTIC_CLOUD_ID")
    api_key = os.getenv("ELASTIC_API_KEY")
    url = first_env("ELASTICSEARCH_URL", "ELASTIC_URL") or "https://localhost:9200"
    username = first_env("ELASTIC_USER", "ELASTIC_USERNAME") or "elastic"
    password = os.getenv("ELASTIC_PASSWORD")
    ca_certs = first_env("ELASTIC_CA_CERTS", "ELASTIC_CA_CERT")

    kwargs = {
        "verify_certs": env_bool("ELASTIC_VERIFY_CERTS", True),
        "request_timeout": float(os.getenv("ELASTIC_REQUEST_TIMEOUT", request_timeout)),
    }
    if ca_certs:
        kwargs["ca_certs"] = ca_certs
    if api_key:
        kwargs["api_key"] = api_key
    elif username and password:
        kwargs["basic_auth"] = (username, password)

    if cloud_id:
        return Elasticsearch(cloud_id=cloud_id, **kwargs)
    return Elasticsearch([url], **kwargs)


def get_elasticsearch_index(default="pubmed_index"):
    return os.getenv("ELASTIC_INDEX", default)


def get_faiss_search_url(default="http://localhost:5000/search"):
    url = first_env("FAISS_SEARCH_URL", "FAISS_URL") or default
    url = url.rstrip("/")
    return url if url.endswith("/search") else f"{url}/search"


def get_faiss_timeout(default=60):
    return float(os.getenv("FAISS_REQUEST_TIMEOUT", default))
