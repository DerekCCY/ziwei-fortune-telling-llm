import os

print("Starting Full Ziwei Chart Pipeline...")

os.system("python gen_births.py --count 1000 --batches 2")
os.system("python gen_charts.py --batches 2")

print("Pipeline Finished.")
