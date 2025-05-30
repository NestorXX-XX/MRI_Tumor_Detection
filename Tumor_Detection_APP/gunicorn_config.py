import multiprocessing
import os

# Server socket
bind = "0.0.0.0:8002"
backlog = 2048

# Worker processes
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = 'sync'
worker_connections = 1000
timeout = 30
keepalive = 2

# Logging
accesslog = '/var/log/gunicorn/access.log'
errorlog = '/var/log/gunicorn/error.log'
loglevel = 'info'

# Process naming
proc_name = 'tumor_detection_app'

# Server mechanics
daemon = True
pidfile = '/var/run/gunicorn/tumor_detection_app.pid'
umask = 0
user = None
group = None
tmp_upload_dir = None

# SSL
keyfile = None
certfile = None

# Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Tumor_Detection_APP.settings') 