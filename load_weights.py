from deepface.basemodels import SFace, Facenet
from deepface.detectors import Yolo, MtCnn, RetinaFace, MediaPipe

def download_weights():
    models={"sface":SFace.load_model,
            "Facenet512":Facenet.load_facenet512d_model
            }   
    
    for model_name, load_model in models.items():
        model=load_model()
        print(f"{model_name} weights downloaded successfully!")


def download_backend_weights():
    backends={"yolov8":Yolo.YoloClient}
    for model_name,backend in backends.items():
        model=backend.build_model(None)
        print(f"{model_name} backend weights downloaded successfully!")

if __name__ == "__main__":
    download_weights()
    download_backend_weights()
