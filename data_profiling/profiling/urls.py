from django.urls import path
from . import views  # Import views correctly
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path("", views.index, name="index"),
    path("home/", views.home, name="home"),
    path(
        "upload/", views.upload_file, name="upload_file"
    ),  # Single entry for upload_file
    path("result/", views.databot_result, name="databot_result"),
    path("upload-dataset/", views.upload_dataset, name="upload_dataset"),
    path("ask/", views.ask_question, name="ask_question"),
    path("datavis_result/", views.datavis_result, name="datavis_result"),
    path(
        "upload-file/", views.uploading_file, name="uploading_file"
    ),  # Correct reference
    path(
        "generate_chart/", views.generate_chart, name="generate_chart"
    ),  # Correct reference
]

# Serve media files during development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
