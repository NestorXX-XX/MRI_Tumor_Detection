import multiprocessing
import os

# Server socket
bind = "0.0.0.0:8002"
backlog = 2048

# Worker processes
workers = 2
worker_class = 'sync'
worker_connections = 1000
timeout = 30
keepalive = 2

# Logging
accesslog = '/home/nestor/MRI_Tumor_Detection/Tumor_Detection_APP/logs/access.log'
errorlog = '/home/nestor/MRI_Tumor_Detection/Tumor_Detection_APP/logs/error.log'
loglevel = 'debug'

# Process naming
proc_name = 'tumor_detection_app'

# Server mechanics
daemon = False
pidfile = '/home/nestor/MRI_Tumor_Detection/Tumor_Detection_APP/logs/tumor_detection_app.pid'
umask = 0
user = None
group = None
tmp_upload_dir = None

# SSL
keyfile = None
certfile = None

# Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Tumor_Detection_APP.settings')
os.environ.setdefault('PYTHONPATH', '/home/nestor/MRI_Tumor_Detection/Tumor_Detection_APP')

# Preload app
preload_app = True

# Capture output
capture_output = True

# Enable stdio inheritance
enable_stdio_inheritance = True

# Worker timeout
graceful_timeout = 30

# Max requests
max_requests = 1000
max_requests_jitter = 50

# Server name
server_name = 'tumor_detection_app'

# Forwarded allow ips
forwarded_allow_ips = '*'

# Chdir
chdir = '/home/nestor/MRI_Tumor_Detection/Tumor_Detection_APP' 