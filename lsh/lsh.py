"""
lsh.py

Algorithms based on 'Mining of Massive Datasets'
http://infolab.stanford.edu/~ullman/mmds/ch3.pdf - Section 3.4

"""

from collections import defaultdict
import multiprocessing as mp
import random
import pyhash
import Levenshtein

from jsonleveldb import JsonLevelDB
from .unionfind import UnionFind

class Signature(object):
    """Signature Base class."""

    def __init__(self, dim):
        self.dim = dim
        self.hashes = self.hash_functions()

    def hash_functions(self):
        """Returns dim different hash functions"""
        pass

    def sign(self, object):
        """Return the signature for object s"""
        pass

class MinHashSignature(Signature):
    """Creates signatures for sets/tuples using minhash."""
    def __init__(self, dim, seeds=None):
        self.dim = dim
        self.seeds = self._set_seeds(seeds)
        self.hasher = pyhash.murmur3_32()
        self.hashes = self._hash_functions()
    
    def _set_seeds(self, seeds):
        """Returns random 32 bit seeds for hash functions"""
        if seeds is not None:
            if len(seeds) != self.dim:
                raise Exception("Seeds length should match dim")
            return seeds
        return [random.getrandbits(32) for i in xrange(self.dim)]

    def _hash_functions(self):
        """Return dim different hash functions"""
        def hash_factory(n):
            return lambda x: self.hasher(x.encode('utf-8'), seed=self.seeds[n])
        return [ hash_factory(_) for _ in range(self.dim) ]

    def sign(self, s):
        """Returns minhash signature for set s"""
        sig = [ float("inf") ] * self.dim
        for hash_ix, hash_fn in enumerate(self.hashes):
            sig[hash_ix] = min(hash_fn(value) for value in s)
        return sig

class LSH(object):
    """Locality sensitive hashing.  Uses a banding approach to hash
    similar signatures to the same buckets."""
    def __init__(self, dim, threshold):
        self.dim = dim
        self.threshold = threshold
        self.bandwidth = self.get_bandwidth(dim, threshold)
        self.hasher = pyhash.murmur3_32()

    def gen_hash(self, sig):
        """Generate hashvals for this signature"""
        for band in zip(*(iter(sig),) * self.bandwidth):
            seed = 0x3456789
            for item in band:
                hashval = self.hasher(str(item), seed=seed)
                seed = hashval
            yield hashval

    def get_bandwidth(self, n, t):
        """Approximates the bandwidth (number of rows in each band)
        needed to get threshold.

        Threshold t = (1/b) ** (1/r) where
        b = #bands
        r = #rows per band
        n = b * r = #elements in signature
        """
        best = n, 1
        minerr  = float("inf")
        for r in range(1, n + 1):
            try:
                b = 1. / (t ** r)
            except:             # Divide by zero, your signature is huge
                return best
            err = abs(n - b * r)
            if err < minerr:
                best = r
                minerr = err
        return best

    def get_threshold(self):
        r = self.bandwidth
        b = self.dim / r
        return (1. / b) ** (1. / r)

    def get_n_bands(self):
        return int(self.dim / self.bandwidth)

class Cluster(object):
    """Clusters sets with Jaccard similarity above threshold with high
    probability.

    Algorithm based on Rajaraman, "Mining of Massive Datasets":
    1. Use LSH to highlight similar signatures as candidate pairs
    2. Verify similarity of candidate pairs with constraint function
    3. Use UnionFind to merge buckets containing same values
    """
    def __init__(self, dim=10, threshold=0.5, shingle_size = None, docs_db = None, state = None):
        self.doccache = dict()
        if state:
            self.dim = state['dim']
            self.threshold = state['threshold']
            self.shingle_size = state['shingle_size']
            self.hasher = LSH(dim, self.threshold)
            self.unionfind = state['unionfind']
            self.hashmaps = state['hashmaps']
            self.labellist = state['labellist']
        else:
            self.dim = dim
            self.threshold = threshold
            self.shingle_size = shingle_size
            self.unionfind = UnionFind()
            self.hasher = LSH(dim, self.threshold)
            self.hashmaps = [defaultdict(list) for _ in range(self.hasher.get_n_bands())]
            self.labellist = set()
        self.docs_db = docs_db #JsonLevelDB documents
        
        #Set up workers
        self.job_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self.children = list()
        for i in range(mp.cpu_count()):
            p = mp.Process(target=self._worker, args=())
            p.daemon = True
            p.start()
            self.children.append(p)

    # Levenshtein similarity computed on candidate pairs
    def _worker(self):
        for label1, label2, band_idx, hshval in iter(self.job_queue.get,"STOP"):
            try:
                text1 = self.doccache[label1]
            except KeyError:
                text1 = self.docs_db.Get(label1)
                self.doccache[label1] = text1
            try:
                text2 = self.doccache[label2]
            except KeyError:
                text2 = self.docs_db.Get(label2)
                self.doccache[label2] = text2

            # For efficiency - skip documents > 10,000 characters
            if len(text1) > 10000 or len(text2) > 10000:
                self.result_queue.put((label1, label2, band_idx, hshval, False))
                continue
            sim = Levenshtein.ratio(text1, text2)
            result = sim > self.threshold
            self.result_queue.put((label1, label2, band_idx, hshval, result))

    def _add_to_unionfind(self, label, sig):
        # Add label to unionfind
        self.unionfind[label]
        self.labellist.add(label)
        
        jobs = 0
        checked = set()
        clustered = dict()
        for band_idx, hshval in enumerate(self.hasher.gen_hash(sig)):
            clustered[(band_idx, hshval)] = False
            for map_label in self.hashmaps[band_idx][hshval]:
                if (label, map_label) not in checked:
                    checked.add((label, map_label))
                    self.job_queue.put((label, map_label, band_idx, hshval))
                    jobs += 1
        
        for i in range(jobs):
            l1, l2, band_idx, hshval, result = self.result_queue.get()
            if result:
                clustered[(band_idx, hshval)] = True
                self.unionfind.union(l1, l2)
                    
        for band_idx, hshval in clustered.iterkeys():
            if not clustered[(band_idx, hshval)]:
                self.hashmaps[band_idx][hshval].append(label)

    def add_signature(self, sig, label):
        """ sig should be a signature tuple/set """
        self._add_to_unionfind(label, sig)
    
    def add_set(self, s, label=None):
        """ s should be a set that defines the item to be clustered """
        if not label:
            label = s
        sig = self._sign(s)
        self._add_to_unionfind(label, sig)

    def get_state(self):
        state = dict()
        state['hashmaps'] = self.hashmaps
        state['dim'] = self.dim
        state['unionfind'] = self.unionfind
        state['threshold'] = self.threshold
        state['labellist'] = self.labellist
        state['shingle_size'] = self.shingle_size
        return state

    def close_workers(self):
        for i in range(mp.cpu_count()):
            self.job_queue.put("STOP")
        for child in self.children:
            child.join()

    def contains(self, label):
        return label in self.labellist

    def get_sets(self):
        return self.unionfind.sets()

    def get_sorted_clusters(self, reverse=True):
        # Returns a list of cluster members (no cluster names)
        clusters = self.get_sets().values()
        clusters.sort(key=len, reverse=reverse)
        return clusters


def shingle(s, k):
    """Generate k-length shingles of string s"""
    k = min(len(s), k)
    for i in range(len(s) - k + 1):
        yield s[i:i+k]

def hshingle(s, k):
    """Generate k-length shingles then hash"""
    for s in shingle(s, k):
        yield hash(s)

def jaccard_sim(X, Y):
    x = set(X)
    y = set(Y)
    """Jaccard similarity between two sets"""
    return float(len(x & y)) / len(x | y)


def jaccard_dist(X, Y):
    """Jaccard distance between two sets"""
    return 1 - jaccard_sim(X, Y)
