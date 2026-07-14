"""Reusable adapter contract suite (package).

Marks ``tests/contract`` as a package so the reusable contract suite is
importable from any adapter test module (tasks 12-16) as::

    from contract.adapter_contract import run_contract_suite

Note: ``tests/`` itself is intentionally NOT a package (existing top-level
test modules are imported by basename). With only this ``__init__.py`` present,
pytest (prepend import mode) inserts ``tests/`` onto ``sys.path`` and the
contract helpers resolve under the top-level ``contract`` package.
"""
