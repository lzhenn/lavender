import sys
CWD=sys.path[0]
sys.path.append(CWD+'/../')

from lavender import toolbox 

output_path=CWD+'/output'
toolbox.flxpt2ply(output_path)