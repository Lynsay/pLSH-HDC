## pLSHHDC : Parallel Locality-Sensitive Hashing based High Dimensional Clustering  

Based on Rajamaran, "Mining of Massive Datasets" - (Section 3.4)[http://infolab.stanford.edu/~ullman/mmds/ch3.pdf]

A parallel implementation of LSH for High Dimensional Clustering.
- Documents or sets are represented by a (MinHash)[http://en.wikipedia.org/wiki/MinHash] signature.
- (LSH)[http://en.wikipedia.org/wiki/Locality-sensitive_hashing] is used to map similar signatures to similar bins.
- Items which map to the same bin are considered candidate pairs for clustering.
- A constraint function (currently Levenshtein distance) is applied to candidate pairs.
- Items which satisfy constraint function are clustered via (UnionFind)[http://en.wikipedia.org/wiki/Disjoint-set_data_structure].

Signatures can be pre-computed (in parallel) and stored using the MinHasher.
Clusters should be built from MinHash signatures.
Constraint checking currently uses the Levenshtein distance of the actual documents stored in a JsonLevelDB database.

Summary of changes:
- Updated to use murmur3 hash (for signatures and LSH)
- Unicode support
- De-coupled clustering from signature creation to allow
  parallel and pre-computation of signatures
- Ability to dump/load signer state to disk
- Constraint function checking for candidate pairs
- Native parallel processing for constraint checks
- Methods to help serialize cluster state to disk

Requires the C-based pyhash, python-Levenshtein, and leveldb libraries

TODO: Remove LevelDB dependency, improve generality of constraint checking.
