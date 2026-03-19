FROM python:3.11

WORKDIR /app

COPY . .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 7860

CMD streamlit run streamlit_app/app.py --server.port 7860 --server.address 0.0.0.0