import grpc
import ml_service_pb2
import ml_service_pb2_grpc
from global_vars import MAX_MESSAGE_LENGTH
from fastapi_service.config import DATA_ROOT, ROOT


def load_data(stub, filepath):
    # Read the .zip file as binary
    with open(filepath, "rb") as f:
        file_data = f.read()

    response = stub.LoadData(
        ml_service_pb2.LoadDataRequest(dataset_file=file_data)
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
    # load_data(stub, DATA_ROOT / 'MNIST.tar.gz')

    # Train model
    # response = stub.TrainModel(
    #     ml_service_pb2.TrainModelRequest(epochs=5,
    #                                      n_classes=10,
    #                                      learning_rate=0.005,
    #                                      batch_size=128,
    #                                      hidden_dim=32,
    #                                      dataset_folder_name='MNIST',
    #                                      model_filename='perceptron_rpc')
    # )

    # print(
    #     "TrainModel Response:",
    #     response.success
    # )

    sample_path = ROOT / 'tests' / 'fixtures' / 'mnist_example3.jpg'

    with open(sample_path, "rb") as f:
        image_file = f.read()

        pred = stub.Predict(
            ml_service_pb2.LoadPredRequest(model_filename='perceptron_rpc',
                                           dataset_folder_name='MNIST',
                                           image_file=image_file,
            )
        )

    print(pred.label, pred.proba)








if __name__ == "__main__":
    run()
