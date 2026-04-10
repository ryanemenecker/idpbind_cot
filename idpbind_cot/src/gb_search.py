import urllib.request
import re

url = "https://raw.githubusercontent.com/openmm/openmm/master/platforms/reference/src/SimTKReference/ReferenceObc.cpp"
try:
    source = urllib.request.urlopen(url).read().decode('utf-8')
    lines = source.split('\n')
    for i, line in enumerate(lines):
        if "offsetRadius" in line or "I =" in line or "H =" in line or "radiusI" in line:
            print(f"{i}: {line}")
except Exception as e:
    print(e)
