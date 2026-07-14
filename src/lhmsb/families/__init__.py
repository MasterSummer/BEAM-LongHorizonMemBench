"""Task content families for LongHorizonMemSysBench.

Each family (e.g. ``software``, ``research``) provides a generator that emits
:class:`~lhmsb.sim.core.FamilyContent` for the simulator core plus a programmatic
:class:`~lhmsb.sim.core.Checker`. Subpackages are imported explicitly by callers
(not re-exported here) so an incomplete or optional family never breaks the rest.
"""
