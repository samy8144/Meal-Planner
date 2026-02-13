print("Checking imports...")
try:
    import matplotlib
    import matplotlib.pyplot as plt
    print("Matplotlib OK")
except Exception as e:
    print(f"Matplotlib Failed: {e}")

try:
    from flask import Flask
    print("Flask OK")
except Exception as e:
    print(f"Flask Failed: {e}")

try:
    import pandas as pd
    print("Pandas OK")
except Exception as e:
    print(f"Pandas Failed: {e}")

try:
    import numpy as np
    print(f"Numpy OK: {np.__version__}")
except Exception as e:
    print(f"Numpy Failed: {e}")

try:
    from sklearn.neighbors import NearestNeighbors
    print("Sklearn Neighbors OK")
except Exception as e:
    print(f"Sklearn Neighbors Failed: {e}")

try:
    import shap
    print(f"SHAP OK: {shap.__version__}")
except Exception as e:
    print(f"SHAP Failed: {e}")
