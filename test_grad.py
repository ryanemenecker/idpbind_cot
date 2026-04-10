import torch
import sys
sys.path.append('.')
from idpbind_cot.tests.test_relaxation_engine import test_obc2_gb_energy_gradients
try:
    test_obc2_gb_energy_gradients()
    print("Test passed successfully!")
except Exception as e:
    print(f"Test failed with error: {e}")
    import traceback
    traceback.print_exc()
