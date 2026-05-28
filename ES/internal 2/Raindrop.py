#normal
from time import sleep
from gpiozero import InputDevice
no_rain = InputDevice(18)

while True:
    if no_rain.is_active:
        print("No rain detected")
    else:
        print("Rain detected")
    sleep(1)