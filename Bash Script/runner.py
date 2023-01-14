import os, json, time

while True:
    parameters= json.load (open('/home/elias/code/model/properties.json', 'r'))
    if parameters["running"] == 0:
        # time.sleep(180)
        print("yes")
        os.system("bash runner.sh")
        
    time.sleep(300)
