# same as raindrop but direct connection
import time
from gpiozero import DigitalInputDevice
mq2 = DigitalInputDevice(17)

while True:
    if mq2.value == 0:
        print("Gas Detected")
    else:
        print('No gas Detected')
    time.sleep(1)