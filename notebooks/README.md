# notebooks/

Exploration only. Nothing importable lives here.

Anything that a backtest depends on belongs in `src/`, where it is covered by the
test suite. Logic that exists only in a notebook cannot be tested for lookahead
bias, and untested logic is how a backtest starts lying.
