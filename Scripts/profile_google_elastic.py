# imports
from elasticsearch import Elasticsearch
import time

es = Elasticsearch(
    'search-taunlp-j3hlsrtqkl2nndztdcypvdf7y4.us-east-1.es.amazonaws.com',
    port=80,
    use_ssl=False,
    verify_certs=False,
    #http_compress = True
)

for iter in range(5):
    batch_size = 20
    res = es.search(index='google_cache', doc_type='results',body={"from": 7200 + iter*200,"size": batch_size * 5,"query": {"match_all":{}}})
    queries = [p['_source']['query'] for p in res['hits']['hits']]

    start = time.time()

    for ind in range(0,len(queries),batch_size):
        res = es.search(index='google_cache', doc_type='results',
                        body={"from": 0, "size": 50, "query": {"bool":{"filter":{"terms":{'query':queries[ind:ind+batch_size]}}}}})
        if res['hits']['total'] != batch_size:
            print('not found')
    start = time.time()
    print("average per sample --- %s seconds ---" % str((time.time() - start)/ len(queries)))

