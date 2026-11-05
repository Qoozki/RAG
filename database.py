# database.py
import jwt
from fastapi import Header, HTTPException, status
from config import SECRET_KEY, ALGORITHM

# User Credentials & Roles Database Mock
USERS_DB = {
    "viewer": {"password": "viewer123", "role": "viewer"},
    "admin": {"password": "admin123", "role": "admin"}
}

def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.PyJWTError:
        return None

def get_current_admin(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing authorization header")
    
    token = authorization.split(" ")[1]
    payload = verify_token(token)
    if not payload or payload.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin privileges required")
    
    return payload
