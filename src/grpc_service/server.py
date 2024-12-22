import io
import tarfile
import time
from concurrent import futures

import grpc
import ml_service_pb2
import ml_service_pb2_grpc

from fastapi_service.config import CLASS_LABELS
from fastapi_service.s3.utils import (
    create_buckets,
    extract_tar_into_s3,
    get_checkpoint_path,
)
from grpc_service.utils import MAX_MESSAGE_LENGTH, FileWrapper
from models import inference, train
from models.modules import LightningPerceptronClassifier


class MLServiceServicer(ml_service_pb2_grpc.MLServiceServicer):
    def LoadData(self, request, context):
        io_dataset_bytes = io.BytesIO(request.dataset_file)
        with tarfile.open(mode="r:gz", fileobj=io_dataset_bytes) as tar:
            extract_tar_into_s3(tar)
        return ml_service_pb2.LoadDataResponse(success=True)

    def TrainModel(self, request, context):
        model = LightningPerceptronClassifier(
            dataset_folder_name=request.dataset_folder_name,
            hidden_dim=request.hidden_dim,
            output_dim=request.n_classes,
            learning_rate=request.learning_rate,
            batch_size=request.batch_size,
        )
        train.train(
            model, request.epochs, request.dataset_folder_name, request.model_filename
        )
        return ml_service_pb2.TrainModelResponse(success=True)

    def Predict(self, request, context):
        model_checkpoint_path = get_checkpoint_path(
            request.dataset_folder_name, request.model_filename
        )
        model = LightningPerceptronClassifier.load_from_checkpoint(
            model_checkpoint_path
        )

        image_file = FileWrapper(io.BytesIO(request.image_file))

        index, value = inference.predict(model, image_file)
        label = str(CLASS_LABELS.get(request.dataset_folder_name)[index])

        return ml_service_pb2.PredictResponse(label=label, probability=value)


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
    # Create if not exists s3 storage buckets on application start up
    create_buckets()
    serve()
