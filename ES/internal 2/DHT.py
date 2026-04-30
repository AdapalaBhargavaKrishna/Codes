# same as IR

import sys
import Adafruit_DHT
import time
while True:
    humidity , temperature = Adafruit_DHT.read_retry(11,3)
    print('Humidity', humidity)
    print('Temperature', temperature)
    time.sleep(1)