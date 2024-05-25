import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or '2e5f89ffa3778921b70a774bc2f65f3f'