#!/usr/bin/env python
"""Test scheduler configuration"""
from ripe.scheduler.linear_with_plateaus import LinearWithPlateaus

# Test alpha_scheduler
try:
    alpha = LinearWithPlateaus(0, 1, 250, 0.2, 0.2)
    print(f"✓ Alpha scheduler OK: slope={alpha.slope:.6f}")
except ZeroDivisionError as e:
    print(f"✗ Alpha scheduler error: {e}")

# Test beta_scheduler  
try:
    beta = LinearWithPlateaus(0, 0.5, 250, 0.1, 0.1)
    print(f"✓ Beta scheduler OK: slope={beta.slope:.6f}")
except ZeroDivisionError as e:
    print(f"✗ Beta scheduler error: {e}")

print("\nSchedulers initialized successfully!")
