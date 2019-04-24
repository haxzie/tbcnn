import sys
import os

if 'test' in sys.argv:
    os.system("classify test tbcnn --in sampler/data/algorithm_trees.pkl --logdir classifier/logs/1 --embed  vectorizer/data/vectors.pkl")
elif 'predict' in sys.argv:
    os.system("python result/predict.py")