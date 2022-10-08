ARG branch

FROM nvcr.io/nvidia/pytorch:21.08-py3 AS base

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install seaborn thop
EXPOSE 3000

COPY . .

FROM base AS client
CMD [ "python", "./client/grpc_testing_client.py" ]
ENV version_name "client"


FROM base AS remote
CMD [ "python", "./remote/grpc_testing_remote.py" ]
ENV version_name "remote"


FROM ${branch} AS final
RUN echo "Built ${version_name} image."
