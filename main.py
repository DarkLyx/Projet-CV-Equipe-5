import sys
from sign_detection import detect_sign, test_luminosity, Mode


options = "lhyr:"
long_options = ["lum", "hsv", "ycrcb", "rgb"]

args = sys.argv
if len(args) > 1:
    arg = args[1]

    if arg in ("--l", long_options[0]):
        test_luminosity()

    elif arg in ("--h", long_options[1]):
        detect_sign(Mode.HSV)

    elif arg in ("--y", long_options[2]):
        detect_sign(Mode.YCRCB)

    elif arg in ("--r", long_options[3]):
        detect_sign(Mode.RGB)

    else :
        print("set to default")
        detect_sign(Mode.RGB)

else :
    print("set to default")
    detect_sign(Mode.RGB)
