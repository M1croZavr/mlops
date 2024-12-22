import io
import tarfile
import time
from concurrent import futures

import grpc
import ml_service_pb2
import ml_service_pb2_grpc
import pytorch_lightning as pl

from fastapi_service.config import CLASS_LABELS
from global_vars import MAX_MESSAGE_LENGTH, FileWrapper
from pytorch_lightning.loggers import MLFlowLogger
from fastapi_service.s3.utils import extract_tar_into_s3, get_checkpoint_path
from models import train, inference

from models.modules import LightningPerceptronClassifier

class MLServiceServicer(ml_service_pb2_grpc.MLServiceServicer):
    def LoadData(self, request, context):

        try:
            io_dataset_bytes = io.BytesIO(request.dataset_file)
            with tarfile.open(mode="r:gz", fileobj=io_dataset_bytes) as tar:
                extract_tar_into_s3(tar)

            message = f"Data extracted into s3"
            success = True
        except Exception as e:
            message = f"Error: {e}"
            success = False

        return ml_service_pb2.LoadDataResponse(success=success, message=message)

    def TrainModel(self, request, context):

        model = LightningPerceptronClassifier(
            dataset_folder_name=request.dataset_folder_name,
            hidden_dim=request.hidden_dim,
            output_dim=request.n_classes,
            learning_rate=request.learning_rate,
            batch_size=request.batch_size,
        )

        train.train(model, request.epochs, request.dataset_folder_name, request.model_filename)

        return ml_service_pb2.TrainModelResponse(
            success=True,
        )

    def Predict(self, request, context):

        model_checkpoint_path = get_checkpoint_path(request.dataset_folder_name, request.model_filename)
        model = LightningPerceptronClassifier.load_from_checkpoint(model_checkpoint_path)

        image_file = FileWrapper(request.image_file)

        index, proba = inference.predict(model, image_file)
        label = CLASS_LABELS.get(request.dataset_folder_name)[index]

        return ml_service_pb2.LoadPredResponse(
            label=label,
            probability=proba
        )



def serve():
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ("grpc.max_message_length", MAX_MESSAGE_LENGTH),
            ("grpc.max_send_message_length", MAX_MESSAGE_LENGTH),
            ("grpc.max_receive_message_length", MAX_MESSAGE_LENGTH),
        ],
    )
    ml_service_pb2_grpc.add_MLServiceServicer_to_server(MLServiceServicer(), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    print("Server started on port 50051")
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == "__main__":
    serve()
