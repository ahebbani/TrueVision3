import requests
print(requests.post("http://localhost:8080/api/mode", json={"mode": "audio"}).status_code)
print(requests.post("http://localhost:8080/api/mode", json={"mode": "face"}).status_code)
print(requests.post("http://localhost:8080/api/mode", json={"mode": "audio"}).status_code)
