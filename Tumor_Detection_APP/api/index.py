from Tumor_Detection_APP.wsgi import application
from django.core.handlers.wsgi import WSGIRequest
from django.http import HttpResponse

def handler(request, event, context):
    if isinstance(request, dict):
        # Convert Vercel request to Django request
        environ = {
            'REQUEST_METHOD': request.get('method', 'GET'),
            'SCRIPT_NAME': '',
            'PATH_INFO': request.get('path', '/'),
            'QUERY_STRING': request.get('query', ''),
            'SERVER_NAME': request.get('headers', {}).get('host', 'localhost'),
            'SERVER_PORT': '80',
            'SERVER_PROTOCOL': 'HTTP/1.1',
            'wsgi.version': (1, 0),
            'wsgi.url_scheme': 'http',
            'wsgi.input': request.get('body', ''),
            'wsgi.errors': None,
            'wsgi.multithread': False,
            'wsgi.multiprocess': False,
            'wsgi.run_once': False,
        }
        request = WSGIRequest(environ)
    
    response = application(request, event, context)
    
    if isinstance(response, HttpResponse):
        return {
            'statusCode': response.status_code,
            'headers': dict(response.headers),
            'body': response.content.decode('utf-8')
        }
    return response 