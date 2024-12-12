import numpy as np
import os
from dataclasses import dataclass
import sys

@dataclass
class PLQ:
    comment: str
    sourceX: float
    sourceY: float
    sourceZ: float

camType = 'plen10' # conv, plen10, ple20
is2D = True #  1D Sensor oder 2D Sensor
sensor_dim = '2D' if is2D else '1D'
isParaxial = True

N_upsample = 1
f_ML = 80

isMLround = True
D_MiL = 320*1e-3
p = D_MiL/57
N_MP_float = D_MiL/p
N_MP = N_MP_float
N_MP_int = int(np.round(N_MP_float))
N_MLA_float = 33
N_MLA: int = int(np.round(N_MLA_float))

f_MLA = 14.2

isMLround = True
p = D_MiL/N_MP_int

numXPixels = N_MLA*N_MP_int
if is2D:
    numYPixels = numXPixels
else:
    numYPixels = 1

D_S = p*numXPixels

p_res = D_S / numXPixels
p_real = D_S / (numXPixels / N_upsample)
if numXPixels > 6000:  # Zemax hat eine maximale Pixelanzahl von 6000
    sys.exit("Zemax hat eine maximale Pixelanzahl von 6000. Ist {}".format(numXPixels))

D_MLA = p*N_MP*N_MLA
isMLAround = False
N_S = N_MLA * N_MP
g_ML_focus =  126
g_ML = np.linspace(start=g_ML_focus-19.5, stop=g_ML_focus+28, num=101)
start_offset = 20
end_offset = 16

b_ML = 1 / (1 / f_ML - 1 / g_ML)

# plen1.0
B_ML_MLA = 1 / (1 / f_ML - 1 / g_ML_focus) # 200#2 * f_ML  # Abstand ML-MLA
g_MLA = B_ML_MLA - b_ML
D_ML = B_ML_MLA * D_MiL / f_MLA # 14.2*4 # 7.3  # Linsendurchmesser ML
b_MLA = 1 / (1 / f_MLA - 1 / g_MLA)
B_ML_Sensor = 1 / (1 / f_ML - 1 / g_ML_focus)
B_MLA_Sensor = f_MLA
lens_pitch_MLA = D_MiL
t_MLA = lens_pitch_MLA * 0.3
r_MLA = 0.5*f_MLA

M_focus=B_ML_MLA/g_ML_focus

# check f-number matching
print("D_MiL/f_MLA=", D_MiL/f_MLA)
print("D_ML/B_ML_MLA=", D_ML/B_ML_MLA)

super_resolution_factor = N_MP_int # N_MP_int 1 3 5 17

if camType == 'plen10':
    plqs = []
    plqs_pos = np.zeros((super_resolution_factor,super_resolution_factor,len(g_ML),3))
    for i_z,z in enumerate(g_ML):
        M = B_ML_MLA / z
        objectspace_spacing = p / M *N_MP_int/super_resolution_factor
        offset = -(N_MP_int // 2) * objectspace_spacing if np.mod(N_MLA, 2) else 0
        for i_y in np.array(range(super_resolution_factor)):
            i_y_temp = i_y-super_resolution_factor//2
            for i_x in np.array(range(super_resolution_factor)):
                i_x_temp = i_x-super_resolution_factor//2
                x = i_x_temp*objectspace_spacing
                y = i_y_temp*objectspace_spacing
                plqs.append(PLQ(str(i_x).zfill(3)+"_"+str(i_y).zfill(3),x,y,z))
                plqs_pos[i_x,i_y,i_z] = [x,y,z]
else:
    plqs = [PLQ(str(0).zfill(3)+"_"+str(0).zfill(3),0,0,0)]

params_path = os.path.abspath(__file__)
