from Tumor_Detection_APP.wsgi import application

def test_wsgi(environ, start_response):
    """Test WSGI application"""
    status = '200 OK'
    headers = [('Content-type', 'text/plain')]
    start_response(status, headers)
    return [b'WSGI is working!'] 