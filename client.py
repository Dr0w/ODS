import requests
import jwt

server_url = "http://localhost:5100/api"

def login(username, password):
    response = requests.post(f"{server_url}/login", json={"username": username, "password": password})
    if response.status_code == 200:
        token = response.json().get("token")
        return token
    else:
        print("Login failed:", response.json())
        return None

def refresh_token(token):
    response = requests.post(f"{server_url}/refresh_token", json={"token": token})
    if response.status_code == 200:
        new_token = response.json().get("token")
        return new_token
    else:
        print("Token refresh failed:", response.json())
        return None

def get_protected_data(token):
    headers = {"Authorization": token}
    response = requests.get(f"{server_url}/protected_route", headers=headers)
    if response.status_code == 200:
        data = response.json()
        print("Protected data:", data)
    else:
        print("Request failed:", response.json())

# Example usage
username = "admin"
password = "admin_password"

token = login(username, password)
if token:
    get_protected_data(token)

    # Refresh token
    new_token = refresh_token(token)
    if new_token:
        get_protected_data(new_token)