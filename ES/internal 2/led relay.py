# gnd 18 23 24 26 vcc
# together
import time
import RPi.GPIO as p
p.setmode(p.BCM)
p.setwarnings(False)

p.setup(27, p.OUT)
p.setup(18, p.OUT)
p.setup(23, p.OUT)
p.setup(24, p.OUT)
p.setup(26, p.OUT)

while True:

    p.output(27, p.LOW)
    p.output(18, p.HIGH)
    p.output(23, p.HIGH)
    p.output(24, p.HIGH)
    p.output(26, p.HIGH)
    print('LED on')
    time.sleep(1)

    p.output(27, p.HIGH)
    p.output(18, p.LOW)
    p.output(23, p.LOW)
    p.output(24, p.LOW)
    p.output(26, p.LOW)
    print('LED off')
    time.sleep(1)