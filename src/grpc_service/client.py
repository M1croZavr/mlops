import grpc
import ml_service_pb2
import ml_service_pb2_grpc
from global_vars import MAX_MESSAGE_LENGTH, MNIST_PATH


def load_data(stub, filepath):
    # Read the .zip file as binary
    with open(filepath, "rb") as f:
        file_data = f.read()

    # Send the file data and filename to the server
    file_name = str(filepath).split("/")[-1]
    response = stub.LoadData(
        ml_service_pb2.LoadDataRequest(file_name=file_name, file_data=file_data)
    )
    print("LoadData Response:", response.message)


def run():
    # Connect to the server
    channel = grpc.insecure_channel(
        "localhost:50051",
        options=[
            ("grpc.max_message_length", MAX_MESSAGE_LENGTH),
            ("grpc.max_send_message_length", MAX_MESSAGE_LENGTH),
            ("grpc.max_receive_message_length", MAX_MESSAGE_LENGTH),
        ],
    )

    stub = ml_service_pb2_grpc.MLServiceStub(channel)

    # Load data by sending .zip file
    load_data(stub, MNIST_PATH)

    # Train model
    response = stub.TrainModel(
        ml_service_pb2.TrainModelRequest(epochs=10, learning_rate=0.001)
    )
    print(
        "TrainModel Response:",
        response.message,
        "Accuracy:",
        response.training_accuracy,
    )


if __name__ == "__main__":
    run()
