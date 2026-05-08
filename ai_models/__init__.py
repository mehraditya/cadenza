"""ai_models — concrete VLA and perception models for Cadenza robots.

Cadenza ships only the framework (base classes, registry, runtime). The
specific models a project uses live here, outside the cadenza package.

Layout::

    ai_models/
        go1/   # VLA, Depth, RGB tuned for the Go1 forward camera
        g1/    # VLA, Depth, RGB tuned for the G1 forward camera
"""
