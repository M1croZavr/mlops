import io
import requests
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

API_URL = "http://127.0.0.1:8000"
FASTAPI_COMMAND = ["poetry", "run", "start-server"]

fastapi_process = None


def is_fastapi_running():
    try:
        response = requests.get(f"{API_URL}/health")
        return response.status_code == 200
    except requests.ConnectionError:
        return False


st.title("Распознователь цифр 3000")
st.write("Данный интерфейс позволяет обучить две нейронки на данных \
          MNIST и затем проверить модель на рукописных цифрах")

st.subheader("1. Загрузка данных")
st.write("Подружаем датасет MNIST")
if st.button("Загрузить данные"):
    with open("data/MNIST_DATA.tar.gz", "rb") as f:
        response = requests.post(f"{API_URL}/mnist/load_data", files={"train_dataset_file": f})
        if response.status_code == 201:
            st.success("MNIST датасет успешно загружен!")
        else:
            st.error("Ошибка при загрузке датасета")


st.subheader("2. Обучение модели")

model_type = st.selectbox("Выберите модель для обучения (можно обучить обе по очереди)", ["Perceptron", "CNN"])
epochs = st.slider("Epochs", 1, 10, 2)
learning_rate = st.slider("Learning Rate", 0.0001, 0.01, 0.001, 0.0001, format='%f')
batch_size = st.slider("Batch Size", 16, 128, 32, 16)

if model_type == "Perceptron":
    hidden_dim = st.slider("Hidden Dimension", 32, 256, 64, 32)
    hyperparameters = {
        "epochs": epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "hidden_dim": hidden_dim
    }
    endpoint = "/mnist/fit_perceptron"
else:
    hidden_channels = st.slider("Hidden Channels", 8, 64, 16, 8)
    hyperparameters = {
        "epochs": epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "hidden_channels": hidden_channels
    }
    endpoint = "/mnist/fit_cnn"


if st.button(f"Обучить {model_type} модель"):
    with st.spinner("Идет обучение..."):
        response = requests.post(f"{API_URL}{endpoint}?model_filename={model_type.lower()}", json=hyperparameters)
    if response.status_code == 201:
        st.success(f"{model_type} модель успешно обучена.")
    else:
        st.error(f"Ошибка при обучении {model_type} модели.")


st.subheader("3. Предсказание цифры на рисунке")

st.write("Нарисуйте цифру")
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=10,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if st.button("Предсказать") and canvas_result.image_data is not None:
    img = Image.fromarray(canvas_result.image_data.astype("uint8")).convert("L")
    img = img.resize((28, 28))

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    img_data = buffer.getvalue()

    perceptron_response = requests.post(
        f"{API_URL}/mnist/predict_perceptron/perceptron",
        files={"file": ("image.jpg", img_data, "image/jpeg")}
    )
    cnn_response = requests.post(
        f"{API_URL}/mnist/predict_cnn/cnn",
        files={"file": ("image.jpg", img_data, "image/jpeg")}
    )

    if perceptron_response.status_code == 200:
        perceptron_result = perceptron_response.json()
        st.write(f"**Perceptron предсказание:** {perceptron_result['label']}")
        st.write(f"**Уверенность:** {perceptron_result['probability']:.2f}")
    else:
        st.error("Ошибка: невозможно предсказать с использованием Perceptron модели")

    if cnn_response.status_code == 200:
        cnn_result = cnn_response.json()
        st.write(f"**CNN предсказание:** {cnn_result['label']}")
        st.write(f"**Уверенность:** {cnn_result['probability']:.2f}")
    else:
        st.error("Ошибка: невозможно предсказать с использованием CNN модели")
