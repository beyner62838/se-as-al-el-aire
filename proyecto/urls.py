"""
Enrutamiento del proyecto.

Define las URLs disponibles:
  /admin/          Panel de administración Django.
  /                Página principal con cámara y reconocimiento en vivo.
  /main/           Alias de la página principal.
  /prediccion/     Entrena (si es necesario) y evalúa el modelo híbrido.
  /camera_predict/ Endpoint API POST para predicción desde la cámara.
"""
from django.contrib import admin
from django.urls import path
from proyecto.views import main
from proyecto.views import prediccion
from proyecto.views import camera_predict
from proyecto.views import sequence_predict

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', main, name='home'),
    path('main/', main, name='main'),
    path('prediccion/', prediccion, name='prediccion'),
    path('camera_predict/', camera_predict, name='camera_predict'),
    path('sequence_predict/', sequence_predict, name='sequence_predict'),
]
