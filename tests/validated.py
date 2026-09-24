"""Our own validated results that later stages and analysis scripts compare against.

Plain constants, no test-framework imports, so analysis scripts can use them without pytest.
Published values from the literature live in tests/published/ instead.
"""

# Stage 5: converged head-to-tail collision edge at the paper energy (r0 = 4000 Richardson
# estimate 5.75459035; r0 = 1000: 5.75459017).
STAGE5_HEAD_TO_TAIL_EDGE = 5.7545902
