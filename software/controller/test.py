import libsensorio
import os

print(libsensorio.core.connect("/dev/"+[x for x in os.listdir("/dev/") if "ttyACM" in x][0]))
libsensorio.core.close()
