
from elasticsearch import Elasticsearch
from config import *
import json
import time
import random
import zlib
import datetime
import base64
import hashlib


es = Elasticsearch(
    'search-taunlp-j3hlsrtqkl2nndztdcypvdf7y4.us-east-1.es.amazonaws.com',
    port=80,
    use_ssl=False,
    verify_certs=False,
    #http_compress = True
)
dbx = dropbox.Dropbox('7j6m2s1jYC0AAAAAAAHy69fu0OxDAU3fPbIjjarqr_1zalj8Mvypf8U71BoLT-AD')

def sync_local_mongo():

    last_sync = '2018-02-20 01:37:14'
    res = es.search(index='google_cache', doc_type='results',
                    body={"from": 0, "size": 50,"aggs": {
                    "max_price": {"max": {"field": "timestamp"}}}})
    last_sync = datetime.datetime.strptime(res['aggregations']['max_price']['value_as_string'], \
                                           "%Y-%m-%dT%H:%M:%S.000Z").strftime('%Y-%m-%d %H:%M:%S')

    file_to_word_on = None
    count=0
    file_count = 0
    bulk_data = []
    folder_list =  dbx.files_list_folder('/cache/')
    while file_count % 2000 == 0:
        for entry in folder_list.entries:
            file_count+=1
            if file_count % 1 ==0:
                print(file_count)
            if entry.client_modified>datetime.datetime.strptime(last_sync,'%Y-%m-%d %H:%M:%S'):
                print(entry.client_modified.strftime('%Y-%m-%d %H:%M:%S'))
                print(entry)
                # copying file to backup and backupdir
                md, res = dbx.files_download('/cache/' + entry.name)
                results = json.loads(res.content)

                for res in results:
                    #dump = json.dumps(res['google_results']).encode('utf-8')
                    #compressed = zlib.compress(dump)
                    #print(len(dump))
                    #print(len(compressed))

                    # Get it back:
                    # obj = cPickle.loads(zlib.decompress(compressed))
                    #obj = json.loads(zlib.decompress(compressed))

                    # updating elastic
                    if 'goog_query' not in res:
                        res['goog_query'] = res['question']

                    new_record = {'results':str(json.dumps(res['google_results'])),'query':res['goog_query'], \
                                  'timestamp':md.client_modified.strftime('%Y-%m-%dT%H:%M:%S.%f'), \
                                  'num_of_results':len(res['google_results'])}

                    m = hashlib.md5()
                    m.update(new_record['query'].encode())
                    id = m.hexdigest()

                    bulk_data.append({
                        "index": {
                            "_index": 'google_cache',
                            "_type": 'results',
                            "_id":id
                        }
                    })

                    bulk_data.append(new_record)

                    if len(bulk_data)>10:
                        res = es.bulk(index='google_cache', body=bulk_data, refresh=True)
                        bulk_data = []
                        if res['errors']:
                            print("error writing logs!!")

                    #if local_db['SearchResults_Cache'].find({'querystr': res['question'], 'type':'SCREEN','page':0 }).count() == 0:
                    #    count += 1
                    #    keys = {'querystr': res['question'], 'last_update': md.client_modified.strftime('%Y-%m-%d %H:%M:%S'),\
                    #                 'count':len(res['google_results']), 'type':'SCREEN','page':0,'results':res['google_results']}
                    #    local_db['SearchResults_Cache'].insert_one(keys)
                print(count)
        if file_count % 2000 == 0:
            folder_list = dbx.files_list_folder_continue(folder_list.cursor)


        # moving the file to backup
        #dbx.files_move('/WebAnswer2/' + entry.name, '/WebAnswer2_synced/' + datetime.datetime.fromtimestamp(time.time()).strftime(
        #                    '%Y-%m-%d_%H_%M_%S') + '__' + entry.name)
    print(file_count)


if __name__ == "__main__":
    sync_local_mongo()

