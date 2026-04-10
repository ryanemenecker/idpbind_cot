import urllib.request

url = "https://raw.githubusercontent.com/openmm/openmm/master/platforms/reference/src/SimTKReference/ReferenceObc.cpp"
try:
    source = urllib.request.urlopen(url).read().decode('utf-8')
    for line in source.split('\n'):
        if "offsetRadiusI" in line or "L =" in line or "U =" in line or "ratio =" in line or "termx =" in line or "log(" in line:
            print(line.strip())
except Exception as e:
    pass

url2 = "https://raw.githubusercontent.com/openmm/openmm/master/openmmapi/src/GBSAOBCForce.cpp"
try:
    source = urllib.request.urlopen(url2).read().decode('utf-8')
    for line in source.split('\n'):
        if "log(" in line:
            print(line.strip())
except:
    pass
