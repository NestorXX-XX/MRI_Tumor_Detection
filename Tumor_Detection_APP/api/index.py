from Tumor_Detection_APP.wsgi import application

def handler(request, event, context):
    return application(request, event, context) 