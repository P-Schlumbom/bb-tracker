import json
import time

v = {}
v['referenceWidth'] = 8
v['outputDim'] = 1170

v['circleMinDist'] = 15
v['circleEdgeStrength'] = 224
v['circleCircularity'] = 18
v['circleMinRadius'] = 5
v['circleMaxRadius'] = 30

v['radius'] = 18

v['radiusOfTheUniverse'] = 30.3

v['markerThreshold'] = 45

v['samplePeriod'] = 600
v['sampleSize'] = 20
v['proximityThreshold'] = 20

v['msPerFrame'] = 40
v['recordingInterval'] = 400

v['record'] = False
v['manipulate'] = False
v['sampling'] = False
v['quickTarget'] = False
v['lockMarkers'] = False

v['nBBs'] = 100

v['focus'] = 120

print(v)

with open('base_files/default_vals','w') as f:
    str_out = json.dumps(v)
    f.write(str_out)

'''for i in range(10):
    print("\rLoading" + "." * i, end=''),
    time.sleep(1)'''