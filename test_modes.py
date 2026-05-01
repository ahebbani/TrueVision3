import requests
import time
print(requests.post("http://localhost:8080/api/mode", json={"mode": "audio"}).json())
time.sleep(2)
print(requests.post("http://localhost:8080/api/mode", json={"mode": "face"}).json())
time.sleep(2)
print(requests.post("http://localhost:8080/api/mode", json={"mode": "audio"}).json())
