import grpc
import ml_service_pb2
import ml_service_pb2_grpc
from utils import MAX_MESSAGE_LENGTH

from fastapi_service.config import DATA_ROOT, ROOT


def run_load_data(stub):
    # Load data for training and validation
    with open(DATA_ROOT / "MNIST.tar.gz", "rb") as file:
        archive_bytes = file.read()
    response = stub.LoadData(ml_service_pb2.LoadDataRequest(dataset_file=archive_bytes))
    print("LoadData Response:", response.success)


def run_train_model(stub):
    # Train the model
    response = stub.TrainModel(
        ml_service_pb2.TrainModelRequest(
            epochs=2,
            n_classes=10,
            learning_rate=0.005,
            batch_size=128,
            hidden_dim=16,
            dataset_folder_name="MNIST",
            model_filename="perceptron_grpc",
        )
    )
    print("TrainModel Response:", response.success)


def run_predict(stub):
    # Predict example by the model
    image_path = ROOT / "tests" / "fixtures" / "mnist_example3.jpg"
    with open(image_path, "rb") as file:
        image_bytes = file.read()
        response = stub.Predict(
            ml_service_pb2.PredictRequest(
                model_filename="perceptron_grpc",
                dataset_folder_name="MNIST",
                image_file=image_bytes,
            )
        )
    print(
        f"Predict Response: label = {response.label} | probability = {response.probability}"
    )


if __name__ == "__main__":
    # Connect to the grpc server
    channel = grpc.insecure_channel(
        "localhost:50051",
        options=[
            ("grpc.max_message_length", MAX_MESSAGE_LENGTH),
            ("grpc.max_send_message_length", MAX_MESSAGE_LENGTH),
            ("grpc.max_receive_message_length", MAX_MESSAGE_LENGTH),
        ],
    )
    stub = ml_service_pb2_grpc.MLServiceStub(channel)

    # run_load_data(stub)
    # run_train_model(stub)
    run_predict(stub)
