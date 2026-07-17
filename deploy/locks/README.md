# Native dependency locks

The four tracked files are bootstrap contracts. The server bootstrap resolves
all transitive dependencies into hash-pinned lockfiles under
`$LHMSB_DATA_ROOT/locks/` and installs them into separate Python 3.11 virtual
environments. The tracked contracts never contain credentials or a host-local
path and are not used as an unverified production lock.
