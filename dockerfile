# FROM python:slim
# COPY . .
# RUN pip install -r requarements.txt
# EXPOSE 8501
# HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
# ENTRYPOINT ["streamlit", "run", "test.py", "--server.port=8501", "--server.address=192.168.31.148"]
# # CMD python -m streamlit run test.py

# app/Dockerfile

# FROM python:3.9-slim

# WORKDIR /app

# RUN apt-get update && apt-get install -y \
#     build-essential \
#     curl \
#     software-properties-common \
#     git \
#     && rm -rf /var/lib/apt/lists/*

# RUN git clone -b nik https://github.com/nikitaopravin/datamind_ai_project.git .
# # https://github.com/nikitaopravin/datamind_ai_project.git
# RUN pip3 install -r requirements.txt

# EXPOSE 8501

# HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# ENTRYPOINT ["streamlit", "run", "test.py", "--server.port=8501"]

FROM python:3.9-slim
EXPOSE 8501
WORKDIR /app
COPY . .
RUN pip3 install -r requirements.txt
CMD streamlit run test.py