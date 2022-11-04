import sys
CWD=sys.path[0]
sys.path.append(CWD+'/../')

from lavender import pipeline

pipeline.waterfall()