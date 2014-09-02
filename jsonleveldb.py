# A wrapper to serialize data read from/written to leveldb in json
import leveldb
import json

class JsonLevelDB(object):
    def __init__(self, filename, **kwargs):
        self._filename = filename
        self._db = leveldb.LevelDB(self._filename, **kwargs)
    
    def Get(self, key, verify_checksums = False, fill_cache = True):
        return json.loads(self._db.Get(key, verify_checksums=verify_checksums, fill_cache=fill_cache))
    
    def Put(self, key, value, sync = False):
        self._db.Put(key, json.dumps(value), sync=sync)
    
    def Delete(self, key, sync = False):
        self._db.Delete(key, sync=sync)

    def Write(self, write_batch, sync = False):
        self._db.Write(write_batch._batch, sync=sync)

    def RangeIter(self, key_from = None, key_to = None, include_value = True, verify_checksums = False, fill_cache = True):
        iterator = self._db.RangeIter(key_from, key_to, include_value=include_value, verify_checksums=verify_checksums, fill_cache=fill_cache) 
        if include_value:
            for k, v in iterator:
                yield k, json.loads(v)
        else:
            for k in iterator:
                yield k

    def GetStats(self):
        return self._db.GetStats()

class JsonWriteBatch(object):
    def __init__(self):
        self._batch = leveldb.WriteBatch()
    
    def Put(self, key, value):
        self._batch.Put(key, json.dumps(value))

    def Delete(self, key):
        self._batch.Delete(key)
