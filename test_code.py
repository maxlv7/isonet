import time

s = time.process_time()

for x in range(1000000):
    pass
e = time.process_time()

print(e-s)