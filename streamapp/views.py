from django.shortcuts import render
from django.http.response import StreamingHttpResponse
from streamapp.camera import VideoCamera

# Create your views here.
camera = VideoCamera()

def index(request):
	context = {}
	if request.POST.get('Update'):
		context = {
			'bool': 'true'
		}
	return render(request, 'streamapp/home.html', context=context)

def gen(camera):
	while True:
		frame = camera.get_frame()
		yield (b'--frame\r\n'
				b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def video_feed(request):
	return StreamingHttpResponse(gen(camera),
					content_type='multipart/x-mixed-replace; boundary=frame')

def update(request):
	camera.update()
	return render(request, 'streamapp/home.html')
