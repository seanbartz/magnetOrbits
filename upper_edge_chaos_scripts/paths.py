"""Repo-relative locations for this study. Import before `physics`: it puts the
repo root on sys.path so the scripts run from any working directory.
Override the folders with UEC_DATA / UEC_OUT."""
import os, sys
HERE=os.path.dirname(os.path.abspath(__file__))
ROOT=os.path.dirname(HERE)
DATA=os.environ.get('UEC_DATA',os.path.join(HERE,'data'))
OUT=os.environ.get('UEC_OUT',os.path.join(HERE,'output'))
if ROOT not in sys.path: sys.path.insert(0,ROOT)
def data(name): os.makedirs(DATA,exist_ok=True); return os.path.join(DATA,name)
def out(name): os.makedirs(OUT,exist_ok=True); return os.path.join(OUT,name)
