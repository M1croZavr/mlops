import io
import pathlib
import tarfile
import time
from concurrent import futures

import grpc
import pytorch_lightning as pl

import ml_service_pb2
import ml_service_pb2_grpc
from global_vars import MAX_MESSAGE_LENGTH, DATA_PATH
from models.modules import LightningPerceptronClassifier


class MLServiceServicer(ml_service_pb2_grpc.MLServiceServicer):
    def LoadData(self, request, context):
        # Save the uploaded .zip file to a local directory
        # file_path = f"./uploads/{request.file_name}"
        # os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Write the received bytes to a .zip file
        # with open(file_path, "wb") as f:
        #     f.write(request.file_data)

        # Extract the .zip file to a folder

        try:
            extract_path = DATA_PATH
            io_train_dataset_bytes = io.BytesIO(request.file_data)
            tar = tarfile.open(mode="r:gz", fileobj=io_train_dataset_bytes)
            tar.extractall(path=DATA_PATH)
            message = f"Data extracted to {extract_path}"
            success = True
        except Exception as e:
            message = f"Error: {e}"
            success = False

        return ml_service_pb2.LoadDataResponse(success=success, message=message)

    def TrainModel(self, request, context):
        print(f"Training model with {request.epochs} epochs and learning rate {request.learning_rate}")
        training_accuracy = 0.85
        success = True

        root = pathlib.Path(__file__).parent.parent.parent

        model = LightningPerceptronClassifier(
            data_root=DATA_PATH / "MNIST_DATA",
            input_dim=28 * 28,
            hidden_dim=16,
            output_dim=10,
            learning_rate=0.001,
            batch_size=32
        )
        # checkpoint_callback = pl.callbacks.ModelCheckpoint(
        #     dirpath=artifacts_dir,
        #     filename=model_filename,
        #     save_top_k=1,
        #     monitor="Validation loss"
        # )
        trainer = pl.Trainer(
            # fast_dev_run=True,
            max_epochs=3,
            enable_checkpointing=False
            # max_epochs=hyperparameters.epochs,
            # default_root_dir=artifacts_dir,
            # callbacks=[checkpoint_callback]
        )
        trainer.fit(model)

        return ml_service_pb2.TrainModelResponse(success=success, message="Training completed",
                                                 training_accuracy=training_accuracy)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10),
                                 options=[('grpc.max_message_length', MAX_MESSAGE_LENGTH),
                                  ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                                  ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH)]
                                 )
    ml_service_pb2_grpc.add_MLServiceServicer_to_server(MLServiceServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Server started on port 50051")
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == "__main__":
    serve()
