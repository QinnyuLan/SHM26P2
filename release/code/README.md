# Release code

`render_blind.py` is the submission entry point. `train_full400.py` is the
reproduction entry point. The `bridge_rgs/` directory is a frozen copy of the
audited renderer and training implementation used by these wrappers.

The wrappers keep all user-facing comments and diagnostics in English. Do not
replace the renderer with a different pixel convention or silently optimize
camera poses during blind inference.
